#!/bin/sh

set -eu

/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-apps/video_pipeline.py \
  "$@"
