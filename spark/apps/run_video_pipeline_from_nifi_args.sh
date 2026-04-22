#!/bin/sh

set -eu

INPUT_VIDEO="${1:-}"
VIDEO_ID="${2:-}"
TRIM_START_SEC="${3:-106}"
EXTRACT_EVERY_N_FRAMES="${4:-10}"

if [ -z "$INPUT_VIDEO" ] || [ -z "$VIDEO_ID" ]; then
  echo "Usage: run_video_pipeline_from_nifi_args.sh <input_video_hdfs_path> <video_id> [trim_start_sec] [extract_every_n_frames]" >&2
  exit 1
fi

/opt/spark/bin/spark-submit \
  --master local[*] \
  --driver-memory 4g \
  --conf spark.driver.maxResultSize=2g \
  --conf spark.executor.memory=2g \
  /opt/spark-apps/video_pipeline.py \
  --input-video "$INPUT_VIDEO" \
  --video-id "$VIDEO_ID" \
  --trim-start-sec "$TRIM_START_SEC" \
  --extract-every-n-frames "$EXTRACT_EVERY_N_FRAMES"
