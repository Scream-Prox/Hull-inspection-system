import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_VIDEO = Path("Sudno_Doc.mp4")
DEFAULT_VIDEO_DIR = Path("nifi/input")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local video to HDFS and start Spark video pipeline in Docker")
    parser.add_argument("--local-video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--local-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--all-from-dir", action="store_true")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--hdfs-video", default=None)
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
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def upload_and_run_for_video(args: argparse.Namespace, local_video: Path) -> None:
    if not local_video.exists():
        raise FileNotFoundError(f"Local video was not found: {local_video}")

    video_id = args.video_id or local_video.stem
    hdfs_video = args.hdfs_video or f"hdfs://namenode:8020/data/raw/videos/{local_video.name}"
    hdfs_video_dir = hdfs_video.rsplit("/", 1)[0]
    temp_container_path = f"/tmp/{local_video.name}"

    run(["docker", "exec", "-i", "namenode", "hdfs", "dfs", "-mkdir", "-p", hdfs_video_dir])
    run(["docker", "cp", str(local_video), f"namenode:{temp_container_path}"])
    run(["docker", "exec", "-i", "namenode", "hdfs", "dfs", "-put", "-f", temp_container_path, hdfs_video])
    run(["docker", "exec", "-i", "namenode", "rm", "-f", temp_container_path])

    spark_submit_command = [
        "docker",
        "exec",
        "-i",
        "nifi",
        "/opt/spark/bin/spark-submit",
        "--master",
        "local[*]",
        "/opt/spark-apps/video_pipeline.py",
        "--input-video",
        hdfs_video,
        "--video-id",
        video_id,
        "--trim-start-sec",
        str(args.trim_start_sec),
        "--extract-every-n-frames",
        str(args.extract_every_n_frames),
        "--min-laplacian-var",
        str(args.min_laplacian_var),
        "--min-brightness",
        str(args.min_brightness),
        "--max-brightness",
        str(args.max_brightness),
        "--hamming-threshold",
        str(args.hamming_threshold),
        "--crop",
        args.crop,
        "--target-width",
        str(args.target_width),
        "--target-height",
        str(args.target_height),
    ]
    if args.max_frames is not None:
        spark_submit_command.extend(["--max-frames", str(args.max_frames)])

    run(spark_submit_command)


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    if args.all_from_dir:
        local_dir = Path(args.local_dir).resolve()
        if not local_dir.exists():
            raise FileNotFoundError(f"Video directory was not found: {local_dir}")
        candidates = sorted(
            path for path in local_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        )
        if not candidates:
            raise RuntimeError(f"No video files were found in directory: {local_dir}")
        return candidates
    return [Path(args.local_video).resolve()]


def main() -> None:
    args = parse_args()
    videos = resolve_videos(args)
    for video in videos:
        print(f"\n=== Processing video: {video.name} ===")
        upload_and_run_for_video(args, video)


if __name__ == "__main__":
    main()
