#!/bin/sh

set -eu

if [ "$#" -lt 2 ]; then
  echo "Usage: run_video_pipeline_from_nifi.sh <input_video_hdfs_path> <video_id> [extra args...]" >&2
  exit 1
fi

INPUT_VIDEO="$1"
VIDEO_ID="$2"
shift 2

/opt/spark/bin/spark-submit \
  --master local[*] \
  --driver-memory 4g \
  --conf spark.driver.maxResultSize=2g \
  --conf spark.executor.memory=2g \
  /opt/spark-apps/video_pipeline.py \
  --input-video "$INPUT_VIDEO" \
  --video-id "$VIDEO_ID" \
  "$@"
