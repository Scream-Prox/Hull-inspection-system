CREATE DATABASE IF NOT EXISTS diploma;

CREATE EXTERNAL TABLE IF NOT EXISTS diploma.video_frames_extracted (
  source_video STRING,
  frame_idx BIGINT,
  timestamp_sec DOUBLE,
  raw_frame_path STRING
)
PARTITIONED BY (video_id STRING)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/data/curated/video_frames_extracted';

CREATE EXTERNAL TABLE IF NOT EXISTS diploma.video_frames_processed (
  source_video STRING,
  fps DOUBLE,
  raw_frame_path STRING,
  frame_idx BIGINT,
  timestamp_sec DOUBLE,
  width INT,
  height INT,
  brightness DOUBLE,
  laplacian_var DOUBLE,
  dhash STRING,
  quality_ok BOOLEAN,
  prev_dhash STRING,
  hamming_to_prev INT,
  is_duplicate BOOLEAN,
  processed_frame_path STRING,
  crop_x1 INT,
  crop_y1 INT,
  crop_x2 INT,
  crop_y2 INT,
  processed_width INT,
  processed_height INT,
  norm_min DOUBLE,
  norm_max DOUBLE,
  norm_mean DOUBLE,
  status STRING,
  error STRING
)
PARTITIONED BY (video_id STRING)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/data/curated/video_frames_processed';

CREATE EXTERNAL TABLE IF NOT EXISTS diploma.video_pipeline_runs (
  video_id STRING,
  source_video STRING,
  trim_start_sec INT,
  extract_every_n_frames INT,
  fps DOUBLE,
  trimmed_frames BIGINT,
  extracted_frames BIGINT,
  quality_ok_frames BIGINT,
  deduplicated_frames BIGINT,
  processed_frames BIGINT,
  segmented_frames BIGINT,
  staging_frames_root STRING,
  processed_frames_root STRING,
  masks_root STRING,
  overlays_root STRING,
  curated_processed_root STRING,
  curated_segmentations_root STRING
)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/data/curated/video_pipeline_runs';

CREATE EXTERNAL TABLE IF NOT EXISTS diploma.video_frame_segmentations (
  source_video STRING,
  frame_idx BIGINT,
  processed_frame_path STRING,
  mask_path STRING,
  overlay_path STRING,
  mask_width INT,
  mask_height INT,
  dominant_class_id INT,
  dominant_class_name STRING,
  predicted_class_ids_csv STRING,
  predicted_class_names_csv STRING,
  class_pixel_counts_json STRING,
  status STRING,
  error STRING
)
PARTITIONED BY (video_id STRING)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/data/curated/video_frame_segmentations';

MSCK REPAIR TABLE diploma.video_frames_extracted;
MSCK REPAIR TABLE diploma.video_frames_processed;
MSCK REPAIR TABLE diploma.video_frame_segmentations;
