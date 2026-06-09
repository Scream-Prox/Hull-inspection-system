
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from segmentation_model import load_model_from_checkpoint, predict_with_loaded_model


RAW_IMAGES_ROOT = "hdfs://namenode:8020/data/raw/images"
PROCESSED_INDEX_ROOT = "hdfs://namenode:8020/data/processed/file_index"
WEB_OUTPUT_ROOT = os.environ.get("WEB_OUTPUT_ROOT", "/opt/web-data/results")
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/models/Model_best.pt")
OVERLAY_ALPHA = float(os.environ.get("PHOTO_OVERLAY_ALPHA", "0.45"))
LATEST_BATCH_WINDOW_SEC = int(os.environ.get("PHOTO_BATCH_WINDOW_SEC", "20"))

CLASS_INFO = [
    (0, "Void", (0, 0, 0)),
    (1, "Normal", (59, 130, 246)),
    (2, "Marine growth", (134, 239, 172)),
    (6, "Paint peel", (252, 165, 165)),
    (9, "Corrosion", (196, 181, 253)),
]
CLASS_NAME_BY_ID = {class_id: class_name for class_id, class_name, _ in CLASS_INFO}
WEB_EXCLUDED_CLASS_IDS = {0}
RISK_WEIGHTS = {
    2: 0.48,
    6: 0.72,
    9: 1.0,
}


def color_rgb_to_hex(color_rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color_rgb)


def decode_image(content: bytes):
    array = np.frombuffer(content, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def image_to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def build_color_mask(pred_mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_id, _, color_rgb in CLASS_INFO:
        mask_rgb[pred_mask == class_id] = color_rgb
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def encode_jpg(image_bgr: np.ndarray) -> bytes:
    ok_encode, encoded = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok_encode:
        raise RuntimeError('Could not encode JPG payload')
    return encoded.tobytes()


def compute_risk_score(class_pixel_ratios: dict[int, float]) -> float:
    weighted = 0.0
    active_classes = 0
    for class_id, ratio in class_pixel_ratios.items():
        if class_id in WEB_EXCLUDED_CLASS_IDS or ratio <= 0:
            continue
        active_classes += 1
        weighted += ratio * RISK_WEIGHTS.get(class_id, 0.25)
    coverage_bonus = min(0.18, active_classes * 0.02)
    corrosion_bonus = min(0.22, class_pixel_ratios.get(9, 0.0) * 0.9)
    defect_bonus = min(0.16, class_pixel_ratios.get(8, 0.0) * 0.8)
    return round(min(0.98, weighted + coverage_bonus + corrosion_bonus + defect_bonus), 3)


def build_benchmark_curve(base_items: int, total_seconds: float, inference_seconds: float) -> list[dict]:
    if base_items <= 0:
        return []
    fixed_overhead = max(0.12, total_seconds * 0.12)
    variable_total = max(0.0, total_seconds - fixed_overhead)
    variable_inference = max(0.0, inference_seconds)
    points = []
    for scale in (1, 2, 3, 4):
        item_count = base_items * scale
        estimated_total = fixed_overhead + variable_total * scale
        estimated_inference = variable_inference * scale
        points.append(
            {
                "scale": scale,
                "items": item_count,
                "pipeline_seconds": round(estimated_total, 3),
                "inference_seconds": round(estimated_inference, 3),
            }
        )
    return points


def export_photo_results(df) -> None:
    pipeline_started = time.perf_counter()
    photo_root = Path(WEB_OUTPUT_ROOT) / 'photos'
    images_root = photo_root / 'images'
    masks_root = photo_root / 'masks'
    overlays_root = photo_root / 'overlays'
    if photo_root.exists():
        shutil.rmtree(photo_root)
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    visible_classes = [
        {
            'class_id': class_id,
            'class_name': class_name,
            'color_rgb': list(color_rgb),
            'color_hex': color_rgb_to_hex(color_rgb),
        }
        for class_id, class_name, color_rgb in CLASS_INFO
        if class_id not in WEB_EXCLUDED_CLASS_IDS
    ]
    class_totals = {
        class_id: {
            'class_id': class_id,
            'class_name': class_name,
            'color_rgb': list(color_rgb),
            'color_hex': color_rgb_to_hex(color_rgb),
            'pixel_count': 0,
            'frames_present': 0,
            'pixel_share': 0.0,
        }
        for class_id, class_name, color_rgb in CLASS_INFO
        if class_id not in WEB_EXCLUDED_CLASS_IDS
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = load_model_from_checkpoint(MODEL_PATH, device=device, num_classes=len(CLASS_INFO))

    images = []
    total_visible_pixels = 0
    inference_wall_seconds = 0.0
    decode_prepare_wall_seconds = 0.0
    postprocess_write_wall_seconds = 0.0
    for row in df.select('path', 'length', 'content').orderBy('modificationTime', ascending=False).toLocalIterator():
        content = bytes(row['content'])
        decode_started = time.perf_counter()
        image = decode_image(content)
        decode_prepare_wall_seconds += time.perf_counter() - decode_started
        if image is None:
            continue

        started = time.perf_counter()
        prediction = predict_with_loaded_model(loaded_model, image_to_tensor(image, device))
        pred_mask = prediction['pred_mask']
        inference_wall_seconds += time.perf_counter() - started

        mask_bgr = build_color_mask(pred_mask)
        overlay_bgr = cv2.addWeighted(image, 1.0 - OVERLAY_ALPHA, mask_bgr, OVERLAY_ALPHA, 0.0)
        postprocess_started = time.perf_counter()
        mask_bytes = encode_jpg(mask_bgr)
        overlay_bytes = encode_jpg(overlay_bgr)

        filename = Path(row['path']).name
        image_path = images_root / filename
        mask_path = masks_root / filename
        overlay_path = overlays_root / filename
        image_path.write_bytes(content)
        mask_path.write_bytes(mask_bytes)
        overlay_path.write_bytes(overlay_bytes)

        class_counts = prediction['class_pixel_counts']
        non_void = {class_id: count for class_id, count in class_counts.items() if class_id != 0 and count > 0}
        dominant_class_id = int(prediction['dominant_class_id'])
        total_pixels = int(pred_mask.size)
        visible_pixels = 0
        for class_id, count in class_counts.items():
            if class_id in WEB_EXCLUDED_CLASS_IDS:
                continue
            class_totals[class_id]['pixel_count'] += count
            if count > 0:
                class_totals[class_id]['frames_present'] += 1
            visible_pixels += count
        total_visible_pixels += visible_pixels

        predicted_class_names = [
            CLASS_NAME_BY_ID[class_id]
            for class_id in prediction['predicted_class_ids']
            if class_id not in WEB_EXCLUDED_CLASS_IDS
        ]

        images.append(
            {
                'filename': filename,
                'hdfs_path': row['path'],
                'size_bytes': int(row['length']),
                'size_kb': round(float(row['length']) / 1024.0, 2),
                'processed_image': f'images/{filename}',
                'mask_image': f'masks/{filename}',
                'overlay_image': f'overlays/{filename}',
                'dominant_class_id': int(dominant_class_id),
                'dominant_class_name': CLASS_NAME_BY_ID[int(dominant_class_id)],
                'predicted_class_names': predicted_class_names,
                'class_pixel_counts': class_counts,
                'class_pixel_ratios': prediction['class_pixel_ratios'],
                'prediction_mode': prediction['prediction_mode'],
                'classification_scores': prediction['classification_scores'],
                'risk_score': compute_risk_score(
                    prediction['class_pixel_ratios']
                ),
            }
        )
        postprocess_write_wall_seconds += time.perf_counter() - postprocess_started

    pipeline_wall_seconds = round(time.perf_counter() - pipeline_started, 4)
    avg_risk = round(sum(item['risk_score'] for item in images) / len(images), 3) if images else 0.0
    max_risk = round(max((item['risk_score'] for item in images), default=0.0), 3)
    risk_band = 'Низкий'
    if avg_risk >= 0.7:
        risk_band = 'Высокий'
    elif avg_risk >= 0.45:
        risk_band = 'Средний'

    if total_visible_pixels > 0:
        for class_id in class_totals:
            class_totals[class_id]['pixel_share'] = round(class_totals[class_id]['pixel_count'] / total_visible_pixels, 6)

    manifest = {
        'result_type': 'photo',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'total_images': len(images),
        'run_counts': {
            'processed_images': len(images),
            'decode_prepare_wall_seconds': round(decode_prepare_wall_seconds, 4),
            'inference_wall_seconds': round(inference_wall_seconds, 4),
            'postprocess_write_wall_seconds': round(postprocess_write_wall_seconds, 4),
            'avg_inference_ms_per_image': round((inference_wall_seconds / len(images)) * 1000.0, 3) if images else 0.0,
            'pipeline_wall_seconds': pipeline_wall_seconds,
            'avg_pipeline_seconds_per_image': round(pipeline_wall_seconds / len(images), 4) if images else 0.0,
        },
        'risk_summary': {
            'score': avg_risk,
            'max_score': max_risk,
            'band': risk_band,
        },
        'benchmark_curve': build_benchmark_curve(len(images), pipeline_wall_seconds, inference_wall_seconds),
        'dynamics_series': [
            {
                'label': item['filename'],
                'index': index + 1,
                'risk_score': item['risk_score'],
            }
            for index, item in enumerate(reversed(images))
        ],
        'visible_classes': visible_classes,
        'class_summaries': sorted(class_totals.values(), key=lambda item: item['pixel_count'], reverse=True),
        'images': images,
    }
    (photo_root / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


spark = SparkSession.builder.appName('DiplomaPreprocess').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

source_df = (
    spark.read.format('binaryFile')
    .load(f'{RAW_IMAGES_ROOT}/*')
    .select('path', 'length', 'content', 'modificationTime')
    .withColumn('filename', F.regexp_extract('path', r'([^/]+)$', 1))
)

source_df.select('path', 'length').show(truncate=False)
source_df.select('path', 'filename', 'length').write.mode('overwrite').parquet(PROCESSED_INDEX_ROOT)

latest_modification = source_df.agg(F.max('modificationTime').alias('latest_modification')).collect()[0]['latest_modification']
if latest_modification is not None:
    latest_batch_df = source_df.filter(
        F.unix_timestamp('modificationTime') >= F.unix_timestamp(F.lit(latest_modification)) - LATEST_BATCH_WINDOW_SEC
    )
    export_photo_results(latest_batch_df)

spark.stop()
