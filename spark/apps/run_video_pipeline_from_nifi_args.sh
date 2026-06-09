#!/bin/sh
set -eu

INPUT_VIDEO="${1:-}"
VIDEO_ID="${2:-}"
TRIM_START_SEC="${3:-106}"
EXTRACT_EVERY_N_FRAMES="${4:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"

if [ -z "$INPUT_VIDEO" ] || [ -z "$VIDEO_ID" ]; then
  echo "Usage: run_video_pipeline_from_nifi_args.sh <input_video_hdfs_path> <video_id> [trim_start_sec] [extract_every_n_frames]" >&2
  exit 1
fi

/opt/spark/bin/spark-submit \
  --master local[2] \
  --driver-memory 2500m \
  --conf spark.driver.maxResultSize=1g \
  --conf spark.executor.memory=1g \
  --conf spark.sql.shuffle.partitions=2 \
  /opt/spark-apps/video_pipeline.py \
  --input-video "$INPUT_VIDEO" \
  --video-id "$VIDEO_ID" \
  --trim-start-sec "$TRIM_START_SEC" \
  --extract-every-n-frames "$EXTRACT_EVERY_N_FRAMES" \
  --target-width 256 \
  --target-height 256
