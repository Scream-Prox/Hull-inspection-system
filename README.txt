Project overview

- `docker-compose.yml` orchestrates HDFS, Spark, Hive and NiFi.
- `webapp/app.py` is the Streamlit UI that shows model results after NiFi and Spark finish processing files from `nifi/input`.
- `spark/Dockerfile` builds the shared Spark runtime with OpenCV and Python dependencies.
- `nifi/Dockerfile` builds the NiFi image with Spark client tools and the same video-processing dependencies.
- `spark/apps/video_pipeline.py` runs the video pipeline on top of HDFS and Spark.
- `spark/apps/segmentation_model.py` restores the trained segmentation and classification model from `Model_best.pt`.
- `run_video_pipeline.py` uploads a local video to HDFS and starts the Spark pipeline inside Docker.
- `ml-service/models/Model_best.pt` stores the trained corrosion, paint peel, and marine growth model used after frame preprocessing.
- `hive/create_video_tables.sql` creates Hive external tables over the curated parquet datasets.

Target architecture

NiFi -> HDFS raw
Spark -> HDFS staging / processed / curated
Hive -> curated parquet tables
Streamlit -> reads dashboard-ready results

Target HDFS layout

- `raw/videos` - source videos
- `staging/frames_raw/<video_id>` - extracted raw frames
- `processed/frames/<video_id>` - processed frames for downstream detection
- `processed/masks/<video_id>` - color segmentation masks from the trained model
- `processed/overlays/<video_id>` - source frame + predicted mask overlay
- `curated/video_frames_extracted` - parquet metadata about extracted frames
- `curated/video_frames_processed` - parquet metadata about cleaned and preprocessed frames
- `curated/video_frame_segmentations` - parquet metadata about masks, overlays and predicted classes
- `curated/video_pipeline_runs` - one run summary per video

Dashboard output layout

- `webapp/results/<video_id>/frames` - processed frames for the web gallery
- `webapp/results/<video_id>/masks` - predicted color masks for the web gallery
- `webapp/results/<video_id>/overlays` - frame + mask overlays for the web gallery
- `webapp/results/<video_id>/manifest.json` - aggregated per-video metrics for Streamlit charts
- `webapp/results/photos/images` - local copies of processed photos for the web gallery
- `webapp/results/photos/masks` - predicted masks for uploaded photos
- `webapp/results/photos/overlays` - photo + mask overlays
- `webapp/results/photos/manifest.json` - aggregated photo results for Streamlit

Video pipeline stages

1. Video trimming and frame extraction
2. Quality filtering and deduplication
3. Preprocessing: crop, resize, normalization stats
4. Semantic segmentation inference with the trained 11-class model
5. Saving processed frames, masks, overlays and parquet metadata to HDFS

Build and start

`docker compose build --no-cache spark-master nifi streamlit-web`
`docker compose up -d`

Open the web UI

`http://localhost:8501`

Run the video pipeline

Default example:
`python run_video_pipeline.py`

Custom example:
`python run_video_pipeline.py --local-video "C:\Visual Studio Code\Diplom\Sudno_Doc.mp4" --video-id sudno_doc --trim-start-sec 106 --extract-every-n-frames 10`

Batch from folder:
`python run_video_pipeline.py --all-from-dir --local-dir "C:\Visual Studio Code\Diplom\nifi\input"`

The launcher will:

- upload the local video into `hdfs://namenode:8020/data/raw/videos/...`
- run `spark-submit` inside the `nifi` container
- write outputs to `staging`, `processed` and `curated`
- load `ml-service/models/Model_best.pt`
- generate masks and overlay images for each processed frame

NiFi mode

If you want the pipeline to start automatically from the NiFi UI, place videos directly into `nifi/input` and use this flow:

`GetFile -> UpdateAttribute -> PutHDFS -> ExecuteStreamCommand`

Recommended NiFi attributes for video files:

- `target_hdfs_dir = /data/raw/videos`
- `video_id = ${filename:substringBeforeLast('.')}`
- `input_video_hdfs = hdfs://namenode:8020${target_hdfs_dir}/${filename}`

`ExecuteStreamCommand` settings:

- `Command Path = /bin/sh`
- `Command Arguments = /opt/spark-apps/run_video_pipeline_from_nifi_args.sh ${input_video_hdfs} ${video_id} 106 10`
- `Ignore STDIN = true`

This keeps the same logic as your photo flow: file appears in `nifi/input`, NiFi sends it to HDFS, then NiFi starts Spark processing automatically.
For the video branch the Spark job runs in `local[*]` mode inside the `nifi` container. This avoids the Python 3.9 vs 3.10 mismatch between the current NiFi image and Spark worker image while keeping the pipeline inside your Docker stack.

Web mode

- open `http://localhost:8501`
- put videos or photos into `nifi/input`
- NiFi picks up the file and starts the corresponding pipeline branch
- after inference the pipeline writes `manifest.json`, frames, masks and overlays into `webapp/results/<video_id>`
- the photo branch writes `webapp/results/photos/manifest.json`, masks, overlays and a local image gallery
- Streamlit automatically lists ready videos, builds charts by the real model classes and also shows processed photos

The Streamlit dashboard is results-only. It does not upload files itself.
Video and photo results both use the segmentation model and include masks and overlays.
For the web dashboard `Void` and `Paint peel` are hidden from charts, as requested.

Useful checks

`docker exec -it namenode hdfs dfs -ls /data/raw/videos`
`docker exec -it namenode hdfs dfs -ls /data/staging/frames_raw`
`docker exec -it namenode hdfs dfs -ls /data/processed/frames`
`docker exec -it namenode hdfs dfs -ls /data/processed/masks`
`docker exec -it namenode hdfs dfs -ls /data/processed/overlays`
`docker exec -it namenode hdfs dfs -ls /data/curated`

Create Hive tables

`docker exec -it hive-server sh -lc "/opt/hive/bin/beeline -u 'jdbc:hive2://' -f /opt/hive/create_video_tables.sql"`

Query Hive data

`docker exec -it hive-server sh -lc "/opt/hive/bin/beeline -u 'jdbc:hive2://' -e 'SHOW TABLES IN diploma; SELECT video_id, frame_idx, processed_frame_path FROM diploma.video_frames_processed LIMIT 20;'"`
`docker exec -it hive-server sh -lc "/opt/hive/bin/beeline -u 'jdbc:hive2://' -e 'SELECT video_id, frame_idx, dominant_class_name, overlay_path FROM diploma.video_frame_segmentations LIMIT 20;'"`
