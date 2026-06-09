#!/bin/bash
set -euo pipefail

export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-/opt/hadoop-config}"

DATA_DIRS=(
  /data/raw/videos
  /data/raw/images
  /data/staging/frames_raw
  /data/processed/frames
  /data/processed/masks
  /data/processed/overlays
  /data/processed/file_index
  /data/curated/video_frames_extracted
  /data/curated/video_frames_processed
  /data/curated/video_frame_segmentations
  /data/curated/video_pipeline_runs
)

wait_for_namenode() {
  until hdfs dfsadmin -safemode get >/dev/null 2>&1; do
    sleep 5
  done
}

leave_safe_mode_if_needed() {
  local state
  state="$(hdfs dfsadmin -safemode get 2>/dev/null || true)"
  if [[ "$state" == *"Safe mode is ON"* ]]; then
    hdfs dfsadmin -safemode leave >/dev/null 2>&1 || true
  fi
}

ensure_hdfs_layout() {
  hdfs dfs -mkdir -p "${DATA_DIRS[@]}" >/dev/null 2>&1 || true
  hdfs dfs -chmod -R 777 /data >/dev/null 2>&1 || true
}

wait_for_namenode
leave_safe_mode_if_needed
ensure_hdfs_layout
