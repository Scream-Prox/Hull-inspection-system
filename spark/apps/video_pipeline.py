import argparse
import json
import os
import shutil
import subprocess
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

from segmentation_model import load_model_from_checkpoint, predict_with_loaded_model


torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)


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
    (1, "Normal", (59, 130, 246)),
    (2, "Marine growth", (134, 239, 172)),
    (6, "Paint peel", (252, 165, 165)),
    (9, "Corrosion", (196, 181, 253)),
]
CLASS_NAME_BY_ID = {class_id: class_name for class_id, class_name, _ in CLASS_INFO}
CLASS_COLOR_BY_ID = {class_id: color for class_id, _, color in CLASS_INFO}
WEB_EXCLUDED_CLASS_IDS = {0}
SIGN_EXCLUDED_CLASS_IDS = {0, 1}
RISK_WEIGHTS = {
    2: 0.48,
    6: 0.72,
    9: 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video -> frames -> dedup -> preprocess on Spark/HDFS")
    parser.add_argument("--input-video", required=True, help="Input video path in HDFS or local FS")
    parser.add_argument("--video-id", default=None, help="Stable video id for HDFS output paths")
    parser.add_argument("--trim-start-sec", type=int, default=106)
    parser.add_argument("--extract-every-n-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=960)
    parser.add_argument("--target-analysis-frames", type=int, default=64)
    parser.add_argument("--coverage-buckets", type=int, default=24)
    parser.add_argument("--frames-per-bucket", type=int, default=3)
    parser.add_argument("--min-laplacian-var", type=float, default=18.0)
    parser.add_argument("--min-brightness", type=float, default=12.0)
    parser.add_argument("--max-brightness", type=float, default=250.0)
    parser.add_argument("--hamming-threshold", type=int, default=1)
    parser.add_argument("--crop", default="0.05,0.10,0.95,0.90")
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--model-path", default="/opt/models/Model_best.pt")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--web-output-root", default=DEFAULT_WEB_OUTPUT_ROOT)
    parser.add_argument("--write-curated", action="store_true", help="Write optional Spark parquet audit tables to HDFS")
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


def probe_video(input_path: Path) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "source_total_frames": total_frames,
        "duration_sec": float(total_frames / fps) if fps > 0 else 0.0,
    }


def extract_frames_with_ffmpeg(
    input_path: Path,
    output_dir: Path,
    start_sec: int,
    fps: float,
    every_n_frames: int,
    max_frames: Optional[int],
    crop: tuple[float, float, float, float],
    target_w: int,
    target_h: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("frame_*.jpg"):
        stale_file.unlink()

    effective_fps = fps if fps and fps > 0 else 25.0
    sample_fps = max(effective_fps / max(every_n_frames, 1), 0.2)
    x1r, y1r, x2r, y2r = crop
    crop_w = max(x2r - x1r, 0.01)
    crop_h = max(y2r - y1r, 0.01)
    vf = (
        f"fps={sample_fps:.6f},"
        f"crop=iw*{crop_w:.6f}:ih*{crop_h:.6f}:iw*{x1r:.6f}:ih*{y1r:.6f},"
        f"scale={target_w}:{target_h}:flags=lanczos"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-q:v",
        "2",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_dir / "frame_%08d.jpg"))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg frame extraction failed: {stderr}") from exc

    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError("ffmpeg did not produce any frames for the selected video segment")
    return frame_paths


def build_extracted_frame_rows(
    local_frame_paths: list[Path],
    output_root: str,
    video_id: str,
    fps: float,
    every_n_frames: int,
) -> list[dict]:
    rows: list[dict] = []
    for ordinal, local_path in enumerate(local_frame_paths):
        frame_idx = ordinal * max(every_n_frames, 1)
        frame_hdfs_path = f"{output_root}/{local_path.name}"
        rows.append(
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "timestamp_sec": float(frame_idx / fps) if fps else None,
                "raw_frame_path": frame_hdfs_path,
            }
        )
    return rows


def upload_selected_staging_frames(spark: SparkSession, processed_rows: list[dict], output_root: str) -> None:
    delete_hdfs_path(spark, output_root)
    ensure_hdfs_dir(spark, output_root)
    written_paths: set[str] = set()
    for row in processed_rows:
        raw_frame_path = str(row["raw_frame_path"])
        if raw_frame_path in written_paths:
            continue
        write_hdfs_bytes(spark, raw_frame_path, bytes(row["processed_content"]))
        written_paths.add(raw_frame_path)


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


def build_processed_frame_record(
    image: np.ndarray,
    image_bytes: bytes,
    raw_frame_path: str,
    processed_frame_path: str,
    video_id: str,
    source_video: str,
    fps: float,
    frame_idx: int,
    timestamp_sec: Optional[float],
    width: int,
    height: int,
    brightness: float,
    laplacian_var: float,
    dhash: str,
    quality_ok: bool,
    prev_dhash: Optional[str],
    hamming_to_prev: Optional[int],
    is_duplicate: bool,
    crop_px: tuple[int, int, int, int],
) -> dict:
    normalized = image.astype(np.float32) / 255.0
    x1, y1, x2, y2 = crop_px
    return {
        "video_id": video_id,
        "source_video": source_video,
        "fps": fps,
        "raw_frame_path": raw_frame_path,
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "width": width,
        "height": height,
        "brightness": brightness,
        "laplacian_var": laplacian_var,
        "dhash": dhash,
        "quality_ok": quality_ok,
        "prev_dhash": prev_dhash,
        "hamming_to_prev": hamming_to_prev,
        "is_duplicate": is_duplicate,
        "processed_frame_path": processed_frame_path,
        "processed_content": image_bytes,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "processed_width": int(image.shape[1]),
        "processed_height": int(image.shape[0]),
        "norm_min": float(normalized.min()),
        "norm_max": float(normalized.max()),
        "norm_mean": float(normalized.mean()),
        "status": "ok",
        "error": None,
    }


def frame_quality_score(row: dict) -> float:
    brightness = float(row.get("brightness") or 0.0)
    laplacian = float(row.get("laplacian_var") or 0.0)
    brightness_balance = max(0.0, 1.0 - abs(brightness - 118.0) / 118.0)
    return laplacian * 0.72 + brightness_balance * 100.0


def visual_distance(left: dict, right: dict) -> int:
    left_hash = str(left.get("dhash") or "")
    right_hash = str(right.get("dhash") or "")
    if not left_hash or not right_hash:
        return 64
    distance = hamming_distance(left_hash, right_hash)
    return int(distance) if distance is not None else 64


def is_scene_diverse(candidate: dict, existing_rows: list[dict], min_distance: int = 8) -> bool:
    if not existing_rows:
        return True
    return all(visual_distance(candidate, existing) >= min_distance for existing in existing_rows)


def select_representative_frames(
    processed_rows: list[dict],
    target_frames: int,
    coverage_buckets: int,
    frames_per_bucket: int,
) -> list[dict]:
    if len(processed_rows) <= max(target_frames, 1):
        return sorted(processed_rows, key=lambda row: row["frame_idx"])

    ordered_rows = sorted(processed_rows, key=lambda row: row["frame_idx"])
    total = len(ordered_rows)
    buckets = max(1, min(coverage_buckets, total))
    bucket_choices: list[list[dict]] = []

    for bucket_idx in range(buckets):
        start = int(bucket_idx * total / buckets)
        end = int((bucket_idx + 1) * total / buckets)
        bucket_rows = ordered_rows[start:end]
        if not bucket_rows:
            continue

        ranked = sorted(
            bucket_rows,
            key=lambda row: (frame_quality_score(row), row["frame_idx"]),
            reverse=True,
        )
        local_choices: list[dict] = []
        for candidate in ranked:
            if any(abs(candidate["frame_idx"] - existing["frame_idx"]) < 6 for existing in local_choices):
                continue
            if not is_scene_diverse(candidate, local_choices, min_distance=8):
                continue
            local_choices.append(candidate)
            if len(local_choices) >= frames_per_bucket:
                break
        if len(local_choices) < frames_per_bucket:
            local_indices = {row["frame_idx"] for row in local_choices}
            for candidate in ranked:
                if candidate["frame_idx"] in local_indices:
                    continue
                if any(abs(candidate["frame_idx"] - existing["frame_idx"]) < 6 for existing in local_choices):
                    continue
                local_choices.append(candidate)
                local_indices.add(candidate["frame_idx"])
                if len(local_choices) >= frames_per_bucket:
                    break
        bucket_choices.append(local_choices or ranked[:1])

    selected: list[dict] = []
    selected_indices: set[int] = set()
    max_bucket_depth = max((len(items) for items in bucket_choices), default=0)
    for depth in range(max_bucket_depth):
        for choices in bucket_choices:
            if depth >= len(choices):
                continue
            candidate = choices[depth]
            if candidate["frame_idx"] in selected_indices:
                continue
            if not is_scene_diverse(candidate, selected, min_distance=6) and len(selected) >= buckets:
                continue
            selected.append(candidate)
            selected_indices.add(candidate["frame_idx"])
            if len(selected) >= target_frames:
                return sorted(selected, key=lambda row: row["frame_idx"])

    if len(selected) < target_frames:
        remaining = [row for row in ordered_rows if row["frame_idx"] not in selected_indices]
        remaining = sorted(remaining, key=lambda row: (frame_quality_score(row), -row["frame_idx"]), reverse=True)
        for candidate in remaining:
            selected.append(candidate)
            selected_indices.add(candidate["frame_idx"])
            if len(selected) >= target_frames:
                break

    return sorted(selected, key=lambda row: row["frame_idx"])


def prepare_frames_locally(
    local_frame_paths: list[Path],
    extracted_rows: list[dict],
    processed_frames_root: str,
    video_id: str,
    source_video: str,
    fps: float,
    min_laplacian_var: float,
    min_brightness: float,
    max_brightness: float,
    hamming_threshold: int,
    crop_px: tuple[int, int, int, int],
    target_analysis_frames: int,
    coverage_buckets: int,
    frames_per_bucket: int,
) -> tuple[list[dict], dict]:
    extracted_by_name = {Path(row["raw_frame_path"]).name: row for row in extracted_rows}
    metrics_rows: list[dict] = []
    processed_rows: list[dict] = []
    quality_ok_frames = 0
    deduplicated_frames = 0
    prev_quality_dhash: Optional[str] = None

    for local_path in local_frame_paths:
        payload = local_path.read_bytes()
        image = decode_image(payload)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dhash = compute_dhash(image)
        width = int(image.shape[1])
        height = int(image.shape[0])

        extracted = extracted_by_name[local_path.name]
        quality_ok = (
            min_brightness <= brightness <= max_brightness
            and laplacian_var >= min_laplacian_var
            and bool(dhash)
        )
        prev_quality_dhash_for_row = prev_quality_dhash
        hamming_to_prev = hamming_distance(dhash, prev_quality_dhash_for_row) if prev_quality_dhash_for_row else None
        is_duplicate = bool(prev_quality_dhash_for_row is not None and hamming_to_prev is not None and hamming_to_prev <= hamming_threshold)
        if quality_ok:
            quality_ok_frames += 1
            prev_quality_dhash = dhash

        metrics_row = {
            "image": image,
            "payload": payload,
            "raw_frame_path": extracted["raw_frame_path"],
            "frame_idx": int(extracted["frame_idx"]),
            "timestamp_sec": extracted.get("timestamp_sec"),
            "brightness": brightness,
            "laplacian_var": laplacian_var,
            "dhash": dhash,
            "width": width,
            "height": height,
            "quality_ok": quality_ok,
            "hamming_to_prev": hamming_to_prev,
            "is_duplicate": is_duplicate,
            "prev_dhash": prev_quality_dhash_for_row if quality_ok else None,
        }
        metrics_rows.append(metrics_row)

        if quality_ok and not is_duplicate:
            deduplicated_frames += 1
            processed_rows.append(
                build_processed_frame_record(
                    image=image,
                    image_bytes=payload,
                    raw_frame_path=extracted["raw_frame_path"],
                    processed_frame_path=f"{processed_frames_root}/{local_path.name}",
                    video_id=video_id,
                    source_video=source_video,
                    fps=fps,
                    frame_idx=int(extracted["frame_idx"]),
                    timestamp_sec=extracted.get("timestamp_sec"),
                    width=width,
                    height=height,
                    brightness=brightness,
                    laplacian_var=laplacian_var,
                    dhash=dhash,
                    quality_ok=quality_ok,
                    prev_dhash=prev_quality_dhash_for_row,
                    hamming_to_prev=hamming_to_prev,
                    is_duplicate=is_duplicate,
                    crop_px=crop_px,
                )
            )

    fallback_mode = "normal"
    if not processed_rows:
        fallback_mode = "raw_extracted"
        fallback_candidates = [row for row in metrics_rows if row["dhash"]][: min(max(len(metrics_rows), 1), 36)]
        for row in fallback_candidates:
            processed_rows.append(
                build_processed_frame_record(
                    image=row["image"],
                    image_bytes=row["payload"],
                    raw_frame_path=row["raw_frame_path"],
                    processed_frame_path=f"{processed_frames_root}/{Path(row['raw_frame_path']).name}",
                    video_id=video_id,
                    source_video=source_video,
                    fps=fps,
                    frame_idx=row["frame_idx"],
                    timestamp_sec=row["timestamp_sec"],
                    width=row["width"],
                    height=row["height"],
                    brightness=row["brightness"],
                    laplacian_var=row["laplacian_var"],
                    dhash=row["dhash"],
                    quality_ok=True,
                    prev_dhash=None,
                    hamming_to_prev=None,
                    is_duplicate=False,
                    crop_px=crop_px,
                )
            )

    candidate_processed_frames = len(processed_rows)
    processed_rows = select_representative_frames(
        processed_rows,
        target_frames=max(target_analysis_frames, 1),
        coverage_buckets=max(coverage_buckets, 1),
        frames_per_bucket=max(frames_per_bucket, 1),
    )

    summary = {
        "quality_ok_frames": quality_ok_frames,
        "deduplicated_frames": deduplicated_frames,
        "candidate_processed_frames": candidate_processed_frames,
        "processed_frames": len(processed_rows),
        "fallback_mode": fallback_mode,
    }
    return processed_rows, summary


def processed_frame_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("video_id", T.StringType(), False),
            T.StructField("source_video", T.StringType(), False),
            T.StructField("fps", T.DoubleType(), True),
            T.StructField("raw_frame_path", T.StringType(), False),
            T.StructField("frame_idx", T.IntegerType(), False),
            T.StructField("timestamp_sec", T.DoubleType(), True),
            T.StructField("width", T.IntegerType(), True),
            T.StructField("height", T.IntegerType(), True),
            T.StructField("brightness", T.DoubleType(), True),
            T.StructField("laplacian_var", T.DoubleType(), True),
            T.StructField("dhash", T.StringType(), True),
            T.StructField("quality_ok", T.BooleanType(), True),
            T.StructField("prev_dhash", T.StringType(), True),
            T.StructField("hamming_to_prev", T.IntegerType(), True),
            T.StructField("is_duplicate", T.BooleanType(), True),
            T.StructField("processed_frame_path", T.StringType(), False),
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


def select_story_frames(
    frame_summaries: list[dict],
    target_frames: int = 4,
    coverage_buckets: int = 4,
) -> list[dict]:
    if len(frame_summaries) <= target_frames:
        return sorted(frame_summaries, key=lambda frame: frame["frame_idx"])

    ordered = sorted(frame_summaries, key=lambda frame: frame["frame_idx"])
    total = len(ordered)
    buckets = max(1, min(coverage_buckets, total))
    selected: list[dict] = []
    selected_keys: set[int] = set()

    for bucket_idx in range(buckets):
        start = int(bucket_idx * total / buckets)
        end = int((bucket_idx + 1) * total / buckets)
        bucket_frames = ordered[start:end]
        if not bucket_frames:
            continue
        best = max(
            bucket_frames,
            key=lambda frame: (
                compute_risk_score(frame["class_pixel_ratios"]),
                len(frame.get("predicted_class_ids", [])),
                frame["frame_idx"],
            ),
        )
        if best["frame_idx"] not in selected_keys:
            selected.append(best)
            selected_keys.add(best["frame_idx"])

    if len(selected) < target_frames:
        remaining = [frame for frame in ordered if frame["frame_idx"] not in selected_keys]
        remaining = sorted(
            remaining,
            key=lambda frame: (
                compute_risk_score(frame["class_pixel_ratios"]),
                len(frame.get("predicted_class_ids", [])),
                -frame["frame_idx"],
            ),
            reverse=True,
        )
        for frame in remaining:
            selected.append(frame)
            selected_keys.add(frame["frame_idx"])
            if len(selected) >= target_frames:
                break

    return sorted(selected[:target_frames], key=lambda frame: frame["frame_idx"])


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
    binary_mode = any(frame.get("prediction_mode") == "binary_segmentation_with_classification" for frame in frame_summaries)
    for frame in frame_summaries:
        if binary_mode:
            score_map = {
                int(class_id): float(score)
                for class_id, score in (frame.get("classification_scores") or {}).items()
            }
            predicted_ids = [int(class_id) for class_id in frame.get("predicted_class_ids", [])]
            sign_ids = [class_id for class_id in predicted_ids if class_id not in SIGN_EXCLUDED_CLASS_IDS]
            if not sign_ids:
                sign_ids = [int(frame.get("primary_sign_id") or 0)] if int(frame.get("primary_sign_id") or 0) not in SIGN_EXCLUDED_CLASS_IDS else []
            for class_id in sign_ids:
                if class_id in WEB_EXCLUDED_CLASS_IDS or class_id not in class_totals:
                    continue
                pseudo_pixels = max(1, int(round(score_map.get(class_id, 0.0) * 1000.0)))
                class_totals[class_id]["pixel_count"] += pseudo_pixels
                class_totals[class_id]["frames_present"] += 1
                total_visible_pixels += pseudo_pixels
            continue

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
                "timestamp_sec": frame.get("timestamp_sec"),
                "dominant_class_id": frame.get("primary_sign_id", frame["dominant_class_id"]),
                "dominant_class_name": frame.get("primary_sign_name", frame["dominant_class_name"]),
                "predicted_class_ids": frame.get("predicted_class_ids", []),
                "class_pixel_ratios": frame["class_pixel_ratios"],
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
                "dominant_class_name": frame.get("primary_sign_name", frame["dominant_class_name"]),
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
        "top_frames": select_story_frames(top_frames, target_frames=6, coverage_buckets=6),
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
    loaded_model = load_model_from_checkpoint(model_path, device=device, num_classes=len(CLASS_INFO))

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
        prediction = predict_with_loaded_model(loaded_model, image_to_tensor(image, device))
        pred_mask = prediction["pred_mask"]
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

        class_counts = prediction["class_pixel_counts"]
        dominant_class_id = int(prediction["dominant_class_id"])
        predicted_ids = [int(class_id) for class_id in prediction["predicted_class_ids"]]
        sign_ids = [class_id for class_id in predicted_ids if class_id not in SIGN_EXCLUDED_CLASS_IDS]
        primary_sign_id = int(sign_ids[0]) if sign_ids else int(dominant_class_id)
        primary_sign_name = CLASS_NAME_BY_ID[int(primary_sign_id)]
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
                "primary_sign_id": primary_sign_id,
                "primary_sign_name": primary_sign_name,
                "predicted_class_ids": predicted_ids,
                "predicted_class_names": [CLASS_NAME_BY_ID[class_id] for class_id in sign_ids],
                "class_pixel_counts": class_counts,
                "class_pixel_ratios": prediction["class_pixel_ratios"],
                "prediction_mode": prediction["prediction_mode"],
                "classification_scores": prediction["classification_scores"],
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
        local_extracted_dir = workspace / "frames"

        extraction_stage_started = time.perf_counter()
        load_video_to_local(spark, args.input_video, local_input_video)
        probe_info = probe_video(local_input_video)
        crop_px = (
            int(probe_info["width"] * crop[0]),
            int(probe_info["height"] * crop[1]),
            int(probe_info["width"] * crop[2]),
            int(probe_info["height"] * crop[3]),
        )
        effective_trim_start = args.trim_start_sec
        if probe_info["duration_sec"] and effective_trim_start >= probe_info["duration_sec"]:
            effective_trim_start = 0

        try:
            local_frame_paths = extract_frames_with_ffmpeg(
                input_path=local_input_video,
                output_dir=local_extracted_dir,
                start_sec=effective_trim_start,
                fps=probe_info["fps"],
                every_n_frames=args.extract_every_n_frames,
                max_frames=args.max_frames,
                crop=crop,
                target_w=args.target_width,
                target_h=args.target_height,
            )
        except RuntimeError as exc:
            if "did not produce any frames" not in str(exc) or effective_trim_start <= 0:
                raise
            # Short videos can legitimately end before the configured trim point.
            # In that case we safely retry from the start instead of failing the whole run.
            effective_trim_start = 0
            local_frame_paths = extract_frames_with_ffmpeg(
                input_path=local_input_video,
                output_dir=local_extracted_dir,
                start_sec=effective_trim_start,
                fps=probe_info["fps"],
                every_n_frames=args.extract_every_n_frames,
                max_frames=args.max_frames,
                crop=crop,
                target_w=args.target_width,
                target_h=args.target_height,
            )
        extracted_rows = build_extracted_frame_rows(
            local_frame_paths=local_frame_paths,
            output_root=staging_video_root,
            video_id=video_id,
            fps=probe_info["fps"],
            every_n_frames=args.extract_every_n_frames,
        )
        extraction_wall_seconds = time.perf_counter() - extraction_stage_started

        if args.write_curated:
            extracted_df = spark.createDataFrame(extracted_rows)
            extracted_df = extracted_df.withColumn("source_video", F.lit(args.input_video))
            extracted_df.write.mode("overwrite").partitionBy("video_id").parquet(CURATED_EXTRACTED_ROOT)

        preprocessing_stage_started = time.perf_counter()
        processed_rows, processing_summary = prepare_frames_locally(
            local_frame_paths=local_frame_paths,
            extracted_rows=extracted_rows,
            processed_frames_root=processed_frames_root,
            video_id=video_id,
            source_video=args.input_video,
            fps=probe_info["fps"],
            min_laplacian_var=args.min_laplacian_var,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
            hamming_threshold=args.hamming_threshold,
            crop_px=crop_px,
            target_analysis_frames=args.target_analysis_frames,
            coverage_buckets=args.coverage_buckets,
            frames_per_bucket=args.frames_per_bucket,
        )
        fallback_mode = processing_summary["fallback_mode"]
        upload_selected_staging_frames(spark, processed_rows, staging_video_root)
        upload_processed_frames(spark, processed_rows, processed_frames_root)
        if args.write_curated:
            processed_ok_df = spark.createDataFrame(
                [{k: v for k, v in row.items() if k != "processed_content"} for row in processed_rows],
                schema=processed_frame_schema(),
            )
            processed_ok_df.write.mode("overwrite").partitionBy("video_id").parquet(CURATED_PROCESSED_ROOT)
        preprocessing_wall_seconds = time.perf_counter() - preprocessing_stage_started

        export_stage_started = time.perf_counter()
        segmentation_rows, frame_summaries, timing_stats = predict_masks_for_frames(
            spark=spark,
            rows=processed_rows,
            model_path=args.model_path,
            masks_root=masks_root,
            overlays_root=overlays_root,
            overlay_alpha=args.overlay_alpha,
            web_output_root=args.web_output_root,
            video_id=video_id,
        )
        if args.write_curated:
            segmentation_df = spark.createDataFrame(segmentation_rows, schema=segmentation_schema())
            segmentation_df.write.mode("overwrite").partitionBy("video_id").parquet(CURATED_SEGMENTATIONS_ROOT)
        export_stage_wall_seconds = time.perf_counter() - export_stage_started

        quality_ok_frames = processing_summary["quality_ok_frames"]
        deduplicated_frames = processing_summary["deduplicated_frames"]
        candidate_processed_frames = processing_summary["candidate_processed_frames"]
        processed_frames = processing_summary["processed_frames"]
        segmented_frames = sum(1 for row in segmentation_rows if row.get("status") == "ok")
        pipeline_wall_seconds = round(time.perf_counter() - pipeline_started, 4)

        run_summary_payload = {
            "video_id": video_id,
            "source_video": args.input_video,
            "trim_start_sec": effective_trim_start,
            "extract_every_n_frames": args.extract_every_n_frames,
            "fps": probe_info["fps"],
            "trimmed_frames": max(int(probe_info["source_total_frames"] - round(effective_trim_start * probe_info["fps"])), 0) if probe_info["fps"] else 0,
            "extracted_frames": len(extracted_rows),
            "quality_ok_frames": quality_ok_frames,
            "deduplicated_frames": deduplicated_frames,
            "candidate_processed_frames": candidate_processed_frames,
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
        if args.write_curated:
            run_summary = spark.createDataFrame([run_summary_payload])
            run_summary.write.mode("overwrite").parquet(f"{CURATED_RUNS_ROOT}/{video_id}")

        build_dashboard_manifest(
            video_id=video_id,
            source_video=args.input_video,
            web_video_root=Path(args.web_output_root) / video_id,
            frame_summaries=frame_summaries,
            run_counts={
                "trimmed_frames": max(int(probe_info["source_total_frames"] - round(args.trim_start_sec * probe_info["fps"])), 0) if probe_info["fps"] else 0,
                "extracted_frames": len(extracted_rows),
                "quality_ok_frames": quality_ok_frames,
                "deduplicated_frames": deduplicated_frames,
                "candidate_processed_frames": candidate_processed_frames,
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
