import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from segmentation_model import load_model_from_checkpoint


STAGING_ROOT = "hdfs://namenode:8020/data/staging/frames_raw"
PROCESSED_FRAMES_ROOT = "hdfs://namenode:8020/data/processed/frames"
MASKS_ROOT = "hdfs://namenode:8020/data/processed/masks"
OVERLAYS_ROOT = "hdfs://namenode:8020/data/processed/overlays"
CURATED_EXTRACTED_ROOT = "hdfs://namenode:8020/data/curated/video_frames_extracted"
CURATED_PROCESSED_ROOT = "hdfs://namenode:8020/data/curated/video_frames_processed"
CURATED_SEGMENTATIONS_ROOT = "hdfs://namenode:8020/data/curated/video_frame_segmentations"
CURATED_RUNS_ROOT = "hdfs://namenode:8020/data/curated/video_pipeline_runs"
DEFAULT_WEB_OUTPUT_ROOT = os.environ.get("WEB_OUTPUT_ROOT", "/opt/web-data/results")

CLASS_INFO = [
    (0, "Void", (0, 0, 0)),
    (1, "Ship hull", (0, 0, 255)),
    (2, "Marine growth", (0, 128, 0)),
    (3, "Anode", (0, 255, 255)),
    (4, "Overboard valve", (64, 224, 208)),
    (5, "Propeller", (128, 0, 128)),
    (6, "Paint peel", (255, 0, 0)),
    (7, "Bilge keel", (255, 165, 0)),
    (8, "Defect", (255, 192, 203)),
    (9, "Corrosion", (255, 255, 0)),
    (10, "Sea chest grating", (255, 182, 193)),
]
CLASS_NAME_BY_ID = {class_id: class_name for class_id, class_name, _ in CLASS_INFO}
CLASS_COLOR_BY_ID = {class_id: color for class_id, _, color in CLASS_INFO}
WEB_EXCLUDED_CLASS_IDS = {0, 6}
RISK_WEIGHTS = {
    1: 0.08,
    2: 0.48,
    3: 0.18,
    4: 0.42,
    5: 0.52,
    7: 0.28,
    8: 0.78,
    9: 1.0,
    10: 0.36,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video -> frames -> dedup -> preprocess on Spark/HDFS")
    parser.add_argument("--input-video", required=True, help="Input video path in HDFS or local FS")
    parser.add_argument("--video-id", default=None, help="Stable video id for HDFS output paths")
    parser.add_argument("--trim-start-sec", type=int, default=106)
    parser.add_argument("--extract-every-n-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-laplacian-var", type=float, default=50.0)
    parser.add_argument("--min-brightness", type=float, default=25.0)
    parser.add_argument("--max-brightness", type=float, default=245.0)
    parser.add_argument("--hamming-threshold", type=int, default=4)
    parser.add_argument("--crop", default="0.05,0.10,0.95,0.90")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=640)
    parser.add_argument("--model-path", default="/opt/models/best_model.pth")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--web-output-root", default=DEFAULT_WEB_OUTPUT_ROOT)
    return parser.parse_args()


def parse_crop(raw_value: str) -> Tuple[float, float, float, float]:
    values = [float(item.strip()) for item in raw_value.split(",")]
    if len(values) != 4:
        raise ValueError("Crop must contain exactly 4 comma-separated floats")
    return tuple(values)  # type: ignore[return-value]


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("DiplomaVideoPipeline")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


def hdfs_fs(spark: SparkSession):
    sc = spark.sparkContext
    return sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())


def hdfs_path(spark: SparkSession, path: str):
    return spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)


def ensure_hdfs_dir(spark: SparkSession, path: str) -> None:
    fs = hdfs_fs(spark)
    fs.mkdirs(hdfs_path(spark, path))


def delete_hdfs_path(spark: SparkSession, path: str) -> None:
    fs = hdfs_fs(spark)
    target = hdfs_path(spark, path)
    if fs.exists(target):
        fs.delete(target, True)


def write_hdfs_bytes(spark: SparkSession, path: str, payload: bytes) -> None:
    fs = hdfs_fs(spark)
    output_stream = fs.create(hdfs_path(spark, path), True)
    try:
        output_stream.write(bytearray(payload))
    finally:
        output_stream.close()


def load_video_to_local(spark: SparkSession, input_video: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if input_video.startswith("hdfs://"):
        jvm = spark.sparkContext._jvm
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        source_fs = hdfs_fs(spark)
        source_path = hdfs_path(spark, input_video)

        if not source_fs.exists(source_path):
            raise RuntimeError(f"Input video was not found in HDFS: {input_video}")

        local_fs = jvm.org.apache.hadoop.fs.FileSystem.getLocal(conf)
        local_target = jvm.org.apache.hadoop.fs.Path(str(local_path))
        jvm.org.apache.hadoop.fs.FileUtil.copy(source_fs, source_path, local_fs, local_target, False, conf)
    else:
        shutil.copyfile(input_video, local_path)

    return local_path.stat().st_size


def trim_video(input_path: Path, output_path: Path, start_sec: int) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(round(start_sec * fps)) if fps else 0
    start_frame = min(start_frame, max(total_frames - 1, 0))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    written_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written_frames += 1

    writer.release()
    cap.release()

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "source_total_frames": total_frames,
        "start_sec": start_sec,
        "start_frame": start_frame,
        "trimmed_frames": written_frames,
    }


def extract_frames_to_hdfs(
    spark: SparkSession,
    video_path: Path,
    output_root: str,
    fps: float,
    every_n_frames: int,
    max_frames: Optional[int],
) -> list[dict]:
    delete_hdfs_path(spark, output_root)
    ensure_hdfs_dir(spark, output_root)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    rows: list[dict] = []
    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            filename = f"frame_{frame_idx:08d}.jpg"
            ok_encode, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok_encode:
                raise RuntimeError(f"Could not encode frame {frame_idx}")
            frame_hdfs_path = f"{output_root}/{filename}"
            write_hdfs_bytes(spark, frame_hdfs_path, encoded.tobytes())
            rows.append(
                {
                    "video_id": output_root.rstrip("/").split("/")[-1],
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": float(frame_idx / fps) if fps else None,
                    "raw_frame_path": frame_hdfs_path,
                }
            )
            saved_count += 1
            if max_frames is not None and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return rows


def compute_dhash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if x else "0" for x in diff.flatten())
    return hex(int(bits, 2))[2:].rjust(16, "0")


def hamming_distance(left: Optional[str], right: Optional[str]) -> Optional[int]:
    if left is None or right is None:
        return None
    left_bits = bin(int(left, 16))[2:].zfill(64)
    right_bits = bin(int(right, 16))[2:].zfill(64)
    return sum(ch1 != ch2 for ch1, ch2 in zip(left_bits, right_bits))


def decode_image(content: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(content, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def metrics_udf():
    schema = T.StructType(
        [
            T.StructField("brightness", T.DoubleType(), True),
            T.StructField("laplacian_var", T.DoubleType(), True),
            T.StructField("dhash", T.StringType(), True),
            T.StructField("width", T.IntegerType(), True),
            T.StructField("height", T.IntegerType(), True),
        ]
    )

    @F.udf(returnType=schema)
    def inner(content: bytes):
        try:
            img = decode_image(content)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return {
                "brightness": float(np.mean(gray)),
                "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "dhash": compute_dhash(img),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        except Exception:
            return None

    return inner


def hamming_udf():
    @F.udf(returnType=T.IntegerType())
    def inner(left: str, right: str):
        return hamming_distance(left, right)

    return inner


def preprocess_udf(crop: tuple[float, float, float, float], target_w: int, target_h: int):
    schema = T.StructType(
        [
            T.StructField("processed_content", T.BinaryType(), True),
            T.StructField("crop_x1", T.IntegerType(), True),
            T.StructField("crop_y1", T.IntegerType(), True),
            T.StructField("crop_x2", T.IntegerType(), True),
            T.StructField("crop_y2", T.IntegerType(), True),
            T.StructField("processed_width", T.IntegerType(), True),
            T.StructField("processed_height", T.IntegerType(), True),
            T.StructField("norm_min", T.DoubleType(), True),
            T.StructField("norm_max", T.DoubleType(), True),
            T.StructField("norm_mean", T.DoubleType(), True),
            T.StructField("status", T.StringType(), True),
            T.StructField("error", T.StringType(), True),
        ]
    )

    @F.udf(returnType=schema)
    def inner(content: bytes):
        try:
            img = decode_image(content)
            if img is None:
                return {
                    "processed_content": None,
                    "crop_x1": None,
                    "crop_y1": None,
                    "crop_x2": None,
                    "crop_y2": None,
                    "processed_width": None,
                    "processed_height": None,
                    "norm_min": None,
                    "norm_max": None,
                    "norm_mean": None,
                    "status": "error",
                    "error": "Could not decode image bytes",
                }

            height, width = img.shape[:2]
            x1r, y1r, x2r, y2r = crop
            x1 = max(0, min(width - 1, int(width * x1r)))
            y1 = max(0, min(height - 1, int(height * y1r)))
            x2 = max(x1 + 1, min(width, int(width * x2r)))
            y2 = max(y1 + 1, min(height, int(height * y2r)))

            cropped = img[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            ok_encode, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok_encode:
                raise RuntimeError("Could not encode processed image")

            return {
                "processed_content": encoded.tobytes(),
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "processed_width": target_w,
                "processed_height": target_h,
                "norm_min": float(normalized.min()),
                "norm_max": float(normalized.max()),
                "norm_mean": float(normalized.mean()),
                "status": "ok",
                "error": None,
            }
        except Exception as exc:
            return {
                "processed_content": None,
                "crop_x1": None,
                "crop_y1": None,
                "crop_x2": None,
                "crop_y2": None,
                "processed_width": None,
                "processed_height": None,
                "norm_min": None,
                "norm_max": None,
                "norm_mean": None,
                "status": "error",
                "error": str(exc),
            }

    return inner


def upload_processed_frames(spark: SparkSession, rows, output_root: str) -> None:
    delete_hdfs_path(spark, output_root)
    ensure_hdfs_dir(spark, output_root)
    for row in rows:
        write_hdfs_bytes(spark, row["processed_frame_path"], bytes(row["processed_content"]))


def segmentation_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("video_id", T.StringType(), False),
            T.StructField("source_video", T.StringType(), True),
            T.StructField("frame_idx", T.LongType(), True),
            T.StructField("processed_frame_path", T.StringType(), True),
            T.StructField("mask_path", T.StringType(), True),
            T.StructField("overlay_path", T.StringType(), True),
            T.StructField("mask_width", T.IntegerType(), True),
            T.StructField("mask_height", T.IntegerType(), True),
            T.StructField("dominant_class_id", T.IntegerType(), True),
            T.StructField("dominant_class_name", T.StringType(), True),
            T.StructField("predicted_class_ids_csv", T.StringType(), True),
            T.StructField("predicted_class_names_csv", T.StringType(), True),
            T.StructField("class_pixel_counts_json", T.StringType(), True),
            T.StructField("status", T.StringType(), True),
            T.StructField("error", T.StringType(), True),
        ]
    )


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
    ok_encode, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok_encode:
        raise RuntimeError("Could not encode JPG payload")
    return encoded.tobytes()


def color_rgb_to_hex(color_rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color_rgb)


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
    fixed_overhead = max(0.25, total_seconds * 0.14)
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


def build_parallel_estimate(base_total_seconds: float) -> list[dict]:
    if base_total_seconds <= 0:
        return []
    return [
        {
            "label": "1 видео",
            "videos": 1,
            "sequential_seconds": round(base_total_seconds, 3),
            "parallel_seconds": round(base_total_seconds, 3),
        },
        {
            "label": "2 видео параллельно",
            "videos": 2,
            "sequential_seconds": round(base_total_seconds * 2, 3),
            "parallel_seconds": round(base_total_seconds * 1.72, 3),
        },
    ]


def prepare_web_output_dirs(web_output_root: str, video_id: str) -> dict[str, Path]:
    video_root = Path(web_output_root) / video_id
    if video_root.exists():
        shutil.rmtree(video_root)

    frames_dir = video_root / "frames"
    masks_dir = video_root / "masks"
    overlays_dir = video_root / "overlays"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    return {
        "video_root": video_root,
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "overlays_dir": overlays_dir,
    }


def build_dashboard_manifest(
    video_id: str,
    source_video: str,
    web_video_root: Path,
    frame_summaries: list[dict],
    run_counts: dict,
    hdfs_roots: dict,
) -> None:
    visible_classes = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "color_rgb": list(color_rgb),
            "color_hex": color_rgb_to_hex(color_rgb),
        }
        for class_id, class_name, color_rgb in CLASS_INFO
        if class_id not in WEB_EXCLUDED_CLASS_IDS
    ]

    class_totals = {
        class_id: {
            "class_id": class_id,
            "class_name": class_name,
            "color_rgb": list(color_rgb),
            "color_hex": color_rgb_to_hex(color_rgb),
            "pixel_count": 0,
            "frames_present": 0,
            "pixel_share": 0.0,
        }
        for class_id, class_name, color_rgb in CLASS_INFO
        if class_id not in WEB_EXCLUDED_CLASS_IDS
    }

    total_visible_pixels = 0
    for frame in frame_summaries:
        class_counts = frame["class_pixel_counts"]
        frame_visible_pixels = 0
        for class_id, count in class_counts.items():
            if class_id in WEB_EXCLUDED_CLASS_IDS:
                continue
            class_totals[class_id]["pixel_count"] += count
            if count > 0:
                class_totals[class_id]["frames_present"] += 1
            frame_visible_pixels += count
        total_visible_pixels += frame_visible_pixels

    if total_visible_pixels > 0:
        for class_id in class_totals:
            class_totals[class_id]["pixel_share"] = round(
                class_totals[class_id]["pixel_count"] / total_visible_pixels,
                6,
            )

    top_frames = []
    dynamics_series = []
    risk_values = []
    for frame in frame_summaries:
        sorted_classes = sorted(
            (
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAME_BY_ID[class_id],
                    "pixel_count": count,
                    "pixel_share": frame["class_pixel_ratios"].get(class_id, 0.0),
                }
                for class_id, count in frame["class_pixel_counts"].items()
                if class_id not in WEB_EXCLUDED_CLASS_IDS and count > 0
            ),
            key=lambda item: item["pixel_count"],
            reverse=True,
        )
        top_frames.append(
            {
                "frame_idx": frame["frame_idx"],
                "dominant_class_id": frame["dominant_class_id"],
                "dominant_class_name": frame["dominant_class_name"],
                "top_classes": sorted_classes[:3],
                "processed_image": frame["processed_image"],
                "mask_image": frame["mask_image"],
                "overlay_image": frame["overlay_image"],
            }
        )
        risk_score = compute_risk_score(frame["class_pixel_ratios"])
        risk_values.append(risk_score)
        dynamics_series.append(
            {
                "frame_idx": frame["frame_idx"],
                "timestamp_sec": frame.get("timestamp_sec"),
                "risk_score": risk_score,
                "dominant_class_name": frame["dominant_class_name"],
            }
        )

    avg_risk = round(sum(risk_values) / len(risk_values), 3) if risk_values else 0.0
    max_risk = round(max(risk_values), 3) if risk_values else 0.0
    risk_band = "Низкий"
    if avg_risk >= 0.7:
        risk_band = "Высокий"
    elif avg_risk >= 0.45:
        risk_band = "Средний"

    manifest = {
        "video_id": video_id,
        "source_video": source_video,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visible_classes": visible_classes,
        "run_counts": run_counts,
        "risk_summary": {
            "score": avg_risk,
            "max_score": max_risk,
            "band": risk_band,
        },
        "benchmark_curve": build_benchmark_curve(
            run_counts.get("segmented_frames", 0),
            run_counts.get("pipeline_wall_seconds", 0.0),
            run_counts.get("inference_wall_seconds", 0.0),
        ),
        "parallel_estimate": build_parallel_estimate(run_counts.get("pipeline_wall_seconds", 0.0)),
        "dynamics_series": dynamics_series,
        "hdfs_roots": hdfs_roots,
        "class_summaries": sorted(class_totals.values(), key=lambda item: item["pixel_count"], reverse=True),
        "frames": frame_summaries,
        "top_frames": top_frames,
    }

    (web_video_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def predict_masks_for_frames(
    spark: SparkSession,
    rows,
    model_path: str,
    masks_root: str,
    overlays_root: str,
    overlay_alpha: float,
    web_output_root: str,
    video_id: str,
) -> tuple[list[dict], list[dict], dict]:
    delete_hdfs_path(spark, masks_root)
    delete_hdfs_path(spark, overlays_root)
    ensure_hdfs_dir(spark, masks_root)
    ensure_hdfs_dir(spark, overlays_root)
    web_dirs = prepare_web_output_dirs(web_output_root, video_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(model_path, device=device, num_classes=len(CLASS_INFO))

    prediction_rows: list[dict] = []
    frame_summaries: list[dict] = []
    inference_wall_seconds = 0.0
    segmented_count = 0
    for row in rows:
        processed_content = row["processed_content"]
        processed_frame_path = row["processed_frame_path"]
        frame_name = Path(processed_frame_path).name
        image = decode_image(bytes(processed_content))
        if image is None:
            prediction_rows.append(
                {
                    "video_id": row["video_id"],
                    "source_video": row["source_video"],
                    "frame_idx": row["frame_idx"],
                    "processed_frame_path": processed_frame_path,
                    "mask_path": None,
                    "overlay_path": None,
                    "mask_width": None,
                    "mask_height": None,
                    "dominant_class_id": None,
                    "dominant_class_name": None,
                    "predicted_class_ids_csv": None,
                    "predicted_class_names_csv": None,
                    "class_pixel_counts_json": None,
                    "status": "error",
                    "error": "Could not decode processed frame bytes",
                }
            )
            continue

        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(image_to_tensor(image, device))
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        inference_wall_seconds += time.perf_counter() - started
        segmented_count += 1

        mask_bgr = build_color_mask(pred_mask)
        overlay_bgr = cv2.addWeighted(image, 1.0 - overlay_alpha, mask_bgr, overlay_alpha, 0.0)
        processed_bytes = bytes(processed_content)
        mask_bytes = encode_jpg(mask_bgr)
        overlay_bytes = encode_jpg(overlay_bgr)

        mask_path = f"{masks_root}/{frame_name}"
        overlay_path = f"{overlays_root}/{frame_name}"
        write_hdfs_bytes(spark, mask_path, mask_bytes)
        write_hdfs_bytes(spark, overlay_path, overlay_bytes)

        unique_ids, counts = np.unique(pred_mask, return_counts=True)
        class_counts = {int(class_id): int(count) for class_id, count in zip(unique_ids, counts)}
        non_void = {class_id: count for class_id, count in class_counts.items() if class_id != 0}
        dominant_class_id = max(non_void, key=non_void.get) if non_void else 0
        predicted_ids = sorted(non_void.keys())
        total_pixels = int(pred_mask.size)

        processed_local_path = web_dirs["frames_dir"] / frame_name
        mask_local_path = web_dirs["masks_dir"] / frame_name
        overlay_local_path = web_dirs["overlays_dir"] / frame_name
        processed_local_path.write_bytes(processed_bytes)
        mask_local_path.write_bytes(mask_bytes)
        overlay_local_path.write_bytes(overlay_bytes)

        prediction_rows.append(
            {
                "video_id": row["video_id"],
                "source_video": row["source_video"],
                "frame_idx": row["frame_idx"],
                "processed_frame_path": processed_frame_path,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "mask_width": int(pred_mask.shape[1]),
                "mask_height": int(pred_mask.shape[0]),
                "dominant_class_id": int(dominant_class_id),
                "dominant_class_name": CLASS_NAME_BY_ID[int(dominant_class_id)],
                "predicted_class_ids_csv": ",".join(str(class_id) for class_id in predicted_ids),
                "predicted_class_names_csv": ",".join(CLASS_NAME_BY_ID[class_id] for class_id in predicted_ids),
                "class_pixel_counts_json": json.dumps(
                    {CLASS_NAME_BY_ID[class_id]: count for class_id, count in sorted(class_counts.items())},
                    ensure_ascii=True,
                ),
                "status": "ok",
                "error": None,
            }
        )
        frame_summaries.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "timestamp_sec": float(row["frame_idx"] / row["fps"]) if row["fps"] else None,
                "dominant_class_id": int(dominant_class_id),
                "dominant_class_name": CLASS_NAME_BY_ID[int(dominant_class_id)],
                "predicted_class_ids": predicted_ids,
                "predicted_class_names": [CLASS_NAME_BY_ID[class_id] for class_id in predicted_ids if class_id not in WEB_EXCLUDED_CLASS_IDS],
                "class_pixel_counts": class_counts,
                "class_pixel_ratios": {
                    class_id: round(count / total_pixels, 6)
                    for class_id, count in class_counts.items()
                },
                "processed_image": str(processed_local_path.relative_to(web_dirs["video_root"])).replace("\\", "/"),
                "mask_image": str(mask_local_path.relative_to(web_dirs["video_root"])).replace("\\", "/"),
                "overlay_image": str(overlay_local_path.relative_to(web_dirs["video_root"])).replace("\\", "/"),
            }
        )

    timing_stats = {
        "inference_wall_seconds": round(inference_wall_seconds, 4),
        "avg_inference_ms_per_frame": round((inference_wall_seconds / segmented_count) * 1000.0, 3) if segmented_count else 0.0,
    }
    return prediction_rows, frame_summaries, timing_stats


def main() -> None:
    pipeline_started = time.perf_counter()
    args = parse_args()
    crop = parse_crop(args.crop)
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    video_name = Path(args.input_video).name
    video_id = args.video_id or Path(video_name).stem

    staging_video_root = f"{STAGING_ROOT}/{video_id}"
    processed_frames_root = f"{PROCESSED_FRAMES_ROOT}/{video_id}"
    masks_root = f"{MASKS_ROOT}/{video_id}"
    overlays_root = f"{OVERLAYS_ROOT}/{video_id}"

    workspace = Path(tempfile.mkdtemp(prefix=f"video-pipeline-{video_id}-"))
    try:
        local_input_video = workspace / video_name
        local_trimmed_video = workspace / f"{Path(video_name).stem}_trimmed.mp4"

        extraction_stage_started = time.perf_counter()
        load_video_to_local(spark, args.input_video, local_input_video)
        trim_info = trim_video(local_input_video, local_trimmed_video, args.trim_start_sec)
        extracted_rows = extract_frames_to_hdfs(
            spark=spark,
            video_path=local_trimmed_video,
            output_root=staging_video_root,
            fps=trim_info["fps"],
            every_n_frames=args.extract_every_n_frames,
            max_frames=args.max_frames,
        )
        extraction_wall_seconds = time.perf_counter() - extraction_stage_started

        extracted_df = spark.createDataFrame(extracted_rows)
        extracted_df = extracted_df.withColumn("source_video", F.lit(args.input_video))
        extracted_df.write.mode("overwrite").partitionBy("video_id").parquet(CURATED_EXTRACTED_ROOT)

        preprocessing_stage_started = time.perf_counter()
        frame_binary_df = (
            spark.read.format("binaryFile").load(f"{staging_video_root}/*.jpg")
            .withColumn("frame_idx", F.regexp_extract("path", r"frame_(\d+)\.jpg", 1).cast("long"))
            .withColumn("video_id", F.lit(video_id))
            .withColumn("source_video", F.lit(args.input_video))
            .withColumn("fps", F.lit(trim_info["fps"]))
            .withColumn("timestamp_sec", F.col("frame_idx") / F.lit(trim_info["fps"]) if trim_info["fps"] else F.lit(None))
        )

        metrics_df = (
            frame_binary_df.withColumn("metrics", metrics_udf()("content"))
            .select(
                "video_id",
                "source_video",
                "fps",
                "path",
                "frame_idx",
                "timestamp_sec",
                "content",
                F.col("metrics.brightness").alias("brightness"),
                F.col("metrics.laplacian_var").alias("laplacian_var"),
                F.col("metrics.dhash").alias("dhash"),
                F.col("metrics.width").alias("width"),
                F.col("metrics.height").alias("height"),
            )
        )

        quality_df = (
            metrics_df.withColumn(
                "quality_ok",
                (
                    F.col("brightness").between(args.min_brightness, args.max_brightness)
                    & (F.col("laplacian_var") >= F.lit(args.min_laplacian_var))
                    & F.col("dhash").isNotNull()
                ),
            )
        )

        window = Window.partitionBy("video_id").orderBy("frame_idx")
        quality_ok_df = quality_df.filter(F.col("quality_ok") == True).cache()

        dedup_df = (
            quality_ok_df
            .withColumn("prev_dhash", F.lag("dhash").over(window))
            .withColumn("hamming_to_prev", hamming_udf()("dhash", "prev_dhash"))
            .withColumn(
                "is_duplicate",
                F.when(F.col("prev_dhash").isNull(), F.lit(False))
                .when(F.col("hamming_to_prev") <= F.lit(args.hamming_threshold), F.lit(True))
                .otherwise(F.lit(False)),
            )
        )

        clean_df = dedup_df.filter(F.col("is_duplicate") == False)

        fallback_mode = "normal"
        processing_source_df = clean_df
        if clean_df.limit(1).count() == 0:
            fallback_mode = "raw_extracted"
            processing_source_df = (
                metrics_df.filter(F.col("dhash").isNotNull())
                .orderBy("frame_idx")
                .limit(min(max(len(extracted_rows), 1), 36))
                .withColumn("quality_ok", F.lit(True))
                .withColumn("prev_dhash", F.lit(None).cast("string"))
                .withColumn("hamming_to_prev", F.lit(None).cast("int"))
                .withColumn("is_duplicate", F.lit(False))
            )

        processed_df = (
            processing_source_df.withColumn("processed", preprocess_udf(crop, args.target_width, args.target_height)("content"))
            .select(
                "video_id",
                "source_video",
                "fps",
                F.col("path").alias("raw_frame_path"),
                "frame_idx",
                "timestamp_sec",
                "width",
                "height",
                "brightness",
                "laplacian_var",
                "dhash",
                "quality_ok",
                "prev_dhash",
                "hamming_to_prev",
                "is_duplicate",
                F.concat(F.lit(f"{processed_frames_root}/"), F.regexp_extract("path", r"([^/]+)$", 1)).alias("processed_frame_path"),
                F.col("processed.processed_content").alias("processed_content"),
                F.col("processed.crop_x1").alias("crop_x1"),
                F.col("processed.crop_y1").alias("crop_y1"),
                F.col("processed.crop_x2").alias("crop_x2"),
                F.col("processed.crop_y2").alias("crop_y2"),
                F.col("processed.processed_width").alias("processed_width"),
                F.col("processed.processed_height").alias("processed_height"),
                F.col("processed.norm_min").alias("norm_min"),
                F.col("processed.norm_max").alias("norm_max"),
                F.col("processed.norm_mean").alias("norm_mean"),
                F.col("processed.status").alias("status"),
                F.col("processed.error").alias("error"),
            )
        )

        processed_ok_df = processed_df.filter(F.col("status") == "ok").cache()
        upload_processed_frames(spark, processed_ok_df.select("processed_frame_path", "processed_content").toLocalIterator(), processed_frames_root)

        processed_ok_df.drop("processed_content").write.mode("overwrite").partitionBy("video_id").parquet(CURATED_PROCESSED_ROOT)
        preprocessing_wall_seconds = time.perf_counter() - preprocessing_stage_started

        export_stage_started = time.perf_counter()
        segmentation_rows, frame_summaries, timing_stats = predict_masks_for_frames(
            spark=spark,
            rows=processed_ok_df.select(
                "video_id",
                "source_video",
                "fps",
                "frame_idx",
                "processed_frame_path",
                "processed_content",
            ).toLocalIterator(),
            model_path=args.model_path,
            masks_root=masks_root,
            overlays_root=overlays_root,
            overlay_alpha=args.overlay_alpha,
            web_output_root=args.web_output_root,
            video_id=video_id,
        )
        segmentation_df = spark.createDataFrame(segmentation_rows, schema=segmentation_schema())
        segmentation_df.write.mode("overwrite").partitionBy("video_id").parquet(CURATED_SEGMENTATIONS_ROOT)
        export_stage_wall_seconds = time.perf_counter() - export_stage_started

        quality_ok_frames = quality_ok_df.count()
        deduplicated_frames = clean_df.count()
        processed_frames = processed_ok_df.count()
        segmented_frames = segmentation_df.filter(F.col("status") == "ok").count()
        pipeline_wall_seconds = round(time.perf_counter() - pipeline_started, 4)

        run_summary_payload = {
            "video_id": video_id,
            "source_video": args.input_video,
            "trim_start_sec": args.trim_start_sec,
            "extract_every_n_frames": args.extract_every_n_frames,
            "fps": trim_info["fps"],
            "trimmed_frames": trim_info["trimmed_frames"],
            "extracted_frames": len(extracted_rows),
            "quality_ok_frames": quality_ok_frames,
            "deduplicated_frames": deduplicated_frames,
            "processed_frames": processed_frames,
            "fallback_mode": fallback_mode,
            "staging_frames_root": staging_video_root,
            "processed_frames_root": processed_frames_root,
            "masks_root": masks_root,
            "overlays_root": overlays_root,
            "curated_processed_root": CURATED_PROCESSED_ROOT,
            "curated_segmentations_root": CURATED_SEGMENTATIONS_ROOT,
            "segmented_frames": segmented_frames,
            "extraction_wall_seconds": round(extraction_wall_seconds, 4),
            "preprocessing_wall_seconds": round(preprocessing_wall_seconds, 4),
            "inference_wall_seconds": timing_stats["inference_wall_seconds"],
            "export_stage_wall_seconds": round(export_stage_wall_seconds, 4),
            "pipeline_wall_seconds": pipeline_wall_seconds,
        }
        run_summary = spark.createDataFrame([run_summary_payload])
        run_summary.write.mode("overwrite").parquet(f"{CURATED_RUNS_ROOT}/{video_id}")

        build_dashboard_manifest(
            video_id=video_id,
            source_video=args.input_video,
            web_video_root=Path(args.web_output_root) / video_id,
            frame_summaries=frame_summaries,
            run_counts={
                "trimmed_frames": trim_info["trimmed_frames"],
                "extracted_frames": len(extracted_rows),
                "quality_ok_frames": quality_ok_frames,
                "deduplicated_frames": deduplicated_frames,
                "processed_frames": processed_frames,
                "segmented_frames": segmented_frames,
                "fallback_mode": fallback_mode,
                "extraction_wall_seconds": round(extraction_wall_seconds, 4),
                "preprocessing_wall_seconds": round(preprocessing_wall_seconds, 4),
                "export_stage_wall_seconds": round(export_stage_wall_seconds, 4),
                "pipeline_wall_seconds": pipeline_wall_seconds,
                "avg_pipeline_seconds_per_segmented_frame": round(pipeline_wall_seconds / segmented_frames, 4) if segmented_frames else 0.0,
                **timing_stats,
            },
            hdfs_roots={
                "staging_frames_root": staging_video_root,
                "processed_frames_root": processed_frames_root,
                "masks_root": masks_root,
                "overlays_root": overlays_root,
                "curated_processed_root": CURATED_PROCESSED_ROOT,
                "curated_segmentations_root": CURATED_SEGMENTATIONS_ROOT,
            },
        )

        print(
            json.dumps(
                {
                    "video_id": video_id,
                    "input_video": args.input_video,
                    "staging_frames_root": staging_video_root,
                    "processed_frames_root": processed_frames_root,
                    "masks_root": masks_root,
                    "overlays_root": overlays_root,
                    "curated_processed_root": CURATED_PROCESSED_ROOT,
                    "curated_segmentations_root": CURATED_SEGMENTATIONS_ROOT,
                    "curated_runs_root": f"{CURATED_RUNS_ROOT}/{video_id}",
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    main()
