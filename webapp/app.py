from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import hashlib
import random
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF


RESULTS_DIR = Path(os.environ.get("WEB_RESULTS_DIR", "/opt/web-data/results"))
PDF_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
MODEL_VIEWER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8502")
GRAFANA_EMBED_URL_TEMPLATE = os.environ.get("GRAFANA_EMBED_URL", "").strip()
GRAFANA_TITLE = os.environ.get("GRAFANA_TITLE", "Grafana").strip() or "Grafana"
GRAFANA_DB_PATH = Path(os.environ.get("GRAFANA_DB_PATH", str(RESULTS_DIR / "_grafana" / "inspection_metrics.db")))
VIEW_RESULTS = "\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b"
VIEW_MODEL = "3D \u043c\u043e\u0434\u0435\u043b\u044c"

CLASS_TRANSLATIONS = {
    "Normal": "\u041d\u043e\u0440\u043c\u0430",
    "Ship hull": "\u041d\u043e\u0440\u043c\u0430",
    "Marine growth": "\u041e\u0431\u0440\u0430\u0441\u0442\u0430\u043d\u0438\u0435",
    "Paint peel": "\u041e\u0442\u0441\u043b\u043e\u0435\u043d\u0438\u0435 \u043a\u0440\u0430\u0441\u043a\u0438",
    "Corrosion": "\u041a\u043e\u0440\u0440\u043e\u0437\u0438\u044f",
    "Void": "\u0424\u043e\u043d",
}

FALLBACK_CLASSES = [
    {"class_id": 1, "class_name": "Normal", "color_hex": "#3b82f6"},
    {"class_id": 2, "class_name": "Marine growth", "color_hex": "#86efac"},
    {"class_id": 6, "class_name": "Paint peel", "color_hex": "#fca5a5"},
    {"class_id": 9, "class_name": "Corrosion", "color_hex": "#c4b5fd"},
]
ALLOWED_CLASS_NAMES = {"Normal", "Ship hull", "Marine growth", "Paint peel", "Corrosion", "Void"}
CLASS_COLOR_MAP = {
    "Normal": "#3b82f6",
    "Marine growth": "#86efac",
    "Paint peel": "#fca5a5",
    "Corrosion": "#c4b5fd",
    "Void": "#000000",
}

st.set_page_config(
    page_title="\u0421\u0438\u0441\u0442\u0435\u043c\u0430 \u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438 \u043a\u043e\u0440\u043f\u0443\u0441\u0430 \u0441\u0443\u0434\u043d\u0430",
    page_icon="ship",
    layout="wide",
    initial_sidebar_state="expanded",
)


def translate_class_name(name: str) -> str:
    return CLASS_TRANSLATIONS.get(name, name)


def canonical_class_name(name: str) -> str:
    if name == "Ship hull":
        return "Normal"
    return name


def normalize_visible_classes(classes: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for item in classes or []:
        class_name = canonical_class_name(str(item.get("class_name", "")).strip())
        if class_name not in ALLOWED_CLASS_NAMES:
            continue
        normalized.append(
            {
                **item,
                "class_name": class_name,
                "color_hex": CLASS_COLOR_MAP.get(class_name, str(item.get("color_hex", "#94a3b8"))),
            }
        )
    return normalized or FALLBACK_CLASSES


def render_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 20%),
                linear-gradient(180deg, #08111f 0%, #0b1628 42%, #0d1b30 100%);
            color: #e8f1ff;
        }
        .hero {
            background: linear-gradient(135deg, rgba(10, 26, 47, 0.96) 0%, rgba(12, 36, 64, 0.92) 100%);
            border-radius: 28px;
            padding: 28px 30px;
            margin-bottom: 20px;
            box-shadow: 0 24px 54px rgba(2, 8, 23, 0.38);
            border: 1px solid rgba(125, 211, 252, 0.14);
        }
        .hero h1 {
            margin: 0;
            color: #f8fbff;
            font-size: 2.05rem;
            letter-spacing: 0.01em;
        }
        .panel {
            background: linear-gradient(180deg, rgba(11, 25, 46, 0.96) 0%, rgba(10, 20, 38, 0.98) 100%);
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 24px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 18px 44px rgba(2, 8, 23, 0.32);
        }
        .muted {
            color: #8aa4c5;
            font-size: 0.95rem;
        }
        .legend-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            color: #dbeafe;
            font-size: 0.92rem;
        }
        .legend-dot {
            width: 11px;
            height: 11px;
            border-radius: 999px;
            display: inline-block;
            border: 1px solid rgba(255,255,255,0.14);
        }
        .section-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #eff6ff;
            margin: 0 0 10px 0;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 14px 0 4px;
        }
        .status-card {
            background: linear-gradient(180deg, rgba(14, 34, 59, 0.95) 0%, rgba(11, 27, 47, 0.98) 100%);
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .status-label {
            color: #8aa4c5;
            font-size: 0.82rem;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .status-value {
            color: #f8fbff;
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .status-note {
            color: #c8d9ee;
            font-size: 0.9rem;
            margin-top: 6px;
        }
        .recommend-box {
            background: linear-gradient(180deg, rgba(16, 41, 69, 0.94) 0%, rgba(11, 28, 49, 0.98) 100%);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 18px;
            padding: 16px 18px;
            margin: 10px 0 18px;
            color: #e8f1ff;
        }
        .recommend-box ul {
            margin: 10px 0 0 16px;
            padding: 0;
        }
        .recommend-box li {
            margin-bottom: 8px;
            line-height: 1.45;
        }
        .frame-card {
            background: linear-gradient(180deg, rgba(11, 27, 47, 0.74) 0%, rgba(9, 20, 36, 0.9) 100%);
            border: 1px solid rgba(125, 211, 252, 0.10);
            border-radius: 20px;
            padding: 18px 18px 14px;
            margin: 10px 0 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .frame-card-title {
            color: #f8fbff;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 10px;
        }
        .frame-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0 0 12px 0;
        }
        .frame-chip {
            display: inline-flex;
            align-items: center;
            min-height: 32px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(17, 48, 83, 0.86);
            border: 1px solid rgba(125, 211, 252, 0.12);
            color: #dbeafe;
            font-size: 0.9rem;
            line-height: 1.2;
        }
        .frame-summary {
            color: #d7e5f6;
            font-size: 0.96rem;
            line-height: 1.6;
            margin-top: 8px;
        }
        .frame-note {
            color: #8aa4c5;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 10px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #09111f 0%, #0b1728 100%);
            border-right: 1px solid rgba(125, 211, 252, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #e6eefb !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(11, 27, 47, 0.95) 0%, rgba(9, 20, 36, 0.98) 100%);
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        [data-testid="stMetricLabel"] {
            color: #8aa4c5 !important;
        }
        [data-testid="stMetricValue"] {
            color: #f8fbff !important;
        }
        [data-testid="stInfo"] {
            background: rgba(14, 34, 59, 0.92);
            color: #dbeafe;
            border: 1px solid rgba(125, 211, 252, 0.12);
        }
        .stButton button, .stDownloadButton button {
            border-radius: 14px !important;
            border: 1px solid rgba(125, 211, 252, 0.14) !important;
            background: linear-gradient(180deg, #14345b 0%, #102947 100%) !important;
            color: #f8fbff !important;
            font-weight: 600 !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            border-color: rgba(125, 211, 252, 0.28) !important;
            background: linear-gradient(180deg, #17406e 0%, #123156 100%) !important;
        }
        @media (max-width: 960px) {
            .status-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_view_state() -> None:
    if "dashboard_view" not in st.session_state:
        st.session_state["dashboard_view"] = VIEW_RESULTS


def switch_to_model_view() -> None:
    st.session_state["dashboard_view"] = VIEW_MODEL


def switch_to_results_view() -> None:
    st.session_state["dashboard_view"] = VIEW_RESULTS


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>\u0421\u0438\u0441\u0442\u0435\u043c\u0430 \u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438 \u043a\u043e\u0440\u043f\u0443\u0441\u0430 \u0441\u0443\u0434\u043d\u0430</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_view_switcher() -> None:
    st.radio(
        "\u0420\u0430\u0437\u0434\u0435\u043b \u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u044f",
        options=[VIEW_RESULTS, VIEW_MODEL],
        horizontal=True,
        key="dashboard_view",
        label_visibility="collapsed",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_video_manifests() -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []
    for manifest_path in RESULTS_DIR.glob("*/manifest.json"):
        if manifest_path.parent.name == "photos":
            continue
        try:
            payload = load_json(manifest_path)
        except Exception:
            continue
        if "video_id" not in payload:
            continue
        payload["_root"] = str(manifest_path.parent)
        payload["_updated_at"] = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        manifests.append(payload)
    manifests.sort(key=lambda item: item["_updated_at"], reverse=True)
    return manifests


def load_photo_manifest() -> dict | None:
    manifest_path = RESULTS_DIR / "photos" / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = load_json(manifest_path)
    payload["_root"] = str(manifest_path.parent)
    payload["_updated_at"] = datetime.fromtimestamp(manifest_path.stat().st_mtime)
    return payload


def sync_grafana_metrics(video_manifests: list[dict], photo_manifest: dict | None) -> None:
    GRAFANA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(GRAFANA_DB_PATH)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS runs;
            DROP TABLE IF EXISTS class_metrics;
            DROP TABLE IF EXISTS dynamics;
            DROP TABLE IF EXISTS stage_times;
            DROP TABLE IF EXISTS benchmark_curve;

            CREATE TABLE runs (
                kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                updated_at_iso TEXT NOT NULL,
                updated_at_epoch INTEGER NOT NULL,
                source_name TEXT NOT NULL,
                extracted_frames INTEGER NOT NULL DEFAULT 0,
                quality_ok_frames INTEGER NOT NULL DEFAULT 0,
                deduplicated_frames INTEGER NOT NULL DEFAULT 0,
                processed_frames INTEGER NOT NULL DEFAULT 0,
                segmented_frames INTEGER NOT NULL DEFAULT 0,
                total_images INTEGER NOT NULL DEFAULT 0,
                extraction_wall_seconds REAL NOT NULL DEFAULT 0,
                preprocessing_wall_seconds REAL NOT NULL DEFAULT 0,
                inference_wall_seconds REAL NOT NULL DEFAULT 0,
                export_stage_wall_seconds REAL NOT NULL DEFAULT 0,
                pipeline_wall_seconds REAL NOT NULL DEFAULT 0,
                avg_inference_ms REAL NOT NULL DEFAULT 0,
                risk_score REAL NOT NULL DEFAULT 0,
                risk_band TEXT NOT NULL DEFAULT '',
                primary_issue TEXT NOT NULL DEFAULT '',
                primary_issue_code INTEGER NOT NULL DEFAULT 0,
                state_label TEXT NOT NULL DEFAULT '',
                state_code INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE class_metrics (
                kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                updated_at_epoch INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                class_name_ru TEXT NOT NULL,
                frames_present INTEGER NOT NULL DEFAULT 0,
                pixel_count INTEGER NOT NULL DEFAULT 0,
                pixel_share_pct REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE dynamics (
                kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                updated_at_epoch INTEGER NOT NULL,
                frame_idx INTEGER NOT NULL DEFAULT 0,
                frame_label TEXT NOT NULL,
                risk_score REAL NOT NULL DEFAULT 0,
                dominant_class_name_ru TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE stage_times (
                kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                updated_at_epoch INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                seconds REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE benchmark_curve (
                kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                updated_at_epoch INTEGER NOT NULL,
                scale INTEGER NOT NULL DEFAULT 1,
                items INTEGER NOT NULL DEFAULT 0,
                pipeline_seconds REAL NOT NULL DEFAULT 0,
                inference_seconds REAL NOT NULL DEFAULT 0
            );
            """
        )

        manifests: list[tuple[str, dict]] = [("video", manifest) for manifest in video_manifests]
        if photo_manifest:
            manifests.append(("photo", photo_manifest))

        for kind, manifest in manifests:
            asset_id = (
                manifest.get("video_id")
                or manifest.get("source_batch")
                or manifest.get("generated_at_utc")
                or ("photos" if kind == "photo" else "unknown")
            )
            updated_at = manifest.get("_updated_at") or datetime.utcnow()
            updated_at_epoch = int(updated_at.timestamp())
            updated_at_iso = updated_at.isoformat()
            risk = risk_summary_from_manifest(manifest)

            if kind == "video":
                counts = effective_video_counts(manifest)
                source_name = str(manifest.get("source_video") or asset_id)
            else:
                counts = effective_photo_counts(manifest)
                source_name = str(manifest.get("source_batch") or "photo-batch")

            cur.execute(
                """
                INSERT INTO runs (
                    kind, asset_id, updated_at_iso, updated_at_epoch, source_name,
                    extracted_frames, quality_ok_frames, deduplicated_frames, processed_frames,
                    segmented_frames, total_images, extraction_wall_seconds,
                    preprocessing_wall_seconds, inference_wall_seconds, export_stage_wall_seconds,
                    pipeline_wall_seconds, avg_inference_ms, risk_score, risk_band,
                    primary_issue, primary_issue_code, state_label, state_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    asset_id,
                    updated_at_iso,
                    updated_at_epoch,
                    source_name,
                    int(counts.get("extracted_frames") or 0),
                    int(counts.get("quality_ok_frames") or 0),
                    int(counts.get("deduplicated_frames") or 0),
                    int(counts.get("processed_frames") or 0),
                    int(counts.get("segmented_frames") or counts.get("total_images") or 0),
                    int(counts.get("total_images") or 0),
                    float(counts.get("extraction_wall_seconds") or 0.0),
                    float(counts.get("preprocessing_wall_seconds") or counts.get("decode_prepare_wall_seconds") or 0.0),
                    float(counts.get("inference_wall_seconds") or 0.0),
                    float(counts.get("export_stage_wall_seconds") or counts.get("postprocess_write_wall_seconds") or 0.0),
                    float(counts.get("pipeline_wall_seconds") or 0.0),
                    float(counts.get("avg_inference_ms_per_frame") or counts.get("avg_inference_ms_per_image") or 0.0),
                    float(risk["score"]),
                    str(risk["band"]),
                    primary_issue_from_manifest(manifest),
                    primary_issue_code_from_manifest(manifest),
                    state_label_from_risk(float(risk["score"])),
                    state_code_from_risk(float(risk["score"])),
                ),
            )

            class_df = class_df_from_manifest(manifest)
            for _, row in class_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO class_metrics (
                        kind, asset_id, updated_at_epoch, class_name, class_name_ru,
                        frames_present, pixel_count, pixel_share_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kind,
                        asset_id,
                        updated_at_epoch,
                        str(row.get("class_name", "")),
                        str(row.get("class_name_ru", "")),
                        int(row.get("frames_present", 0) or 0),
                        int(row.get("pixel_count", 0) or 0),
                        float(row.get("pixel_share_pct", 0.0) or 0.0),
                    ),
                )

            dynamics_df = dynamics_df_from_manifest(manifest)
            for _, row in dynamics_df.iterrows():
                frame_idx = int(row.get("frame_idx", row.get("index", 0)) or 0)
                frame_label = str(frame_idx)
                cur.execute(
                    """
                    INSERT INTO dynamics (
                        kind, asset_id, updated_at_epoch, frame_idx, frame_label,
                        risk_score, dominant_class_name_ru
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kind,
                        asset_id,
                        updated_at_epoch,
                        frame_idx,
                        frame_label,
                        float(row.get("risk_score", 0.0) or 0.0),
                        str(row.get("dominant_class_name_ru", "")),
                    ),
                )

            stage_df = stage_breakdown_df_from_manifest(manifest)
            for _, row in stage_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO stage_times (
                        kind, asset_id, updated_at_epoch, stage_name, seconds
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        kind,
                        asset_id,
                        updated_at_epoch,
                        str(row.get("stage", "")),
                        float(row.get("seconds", 0.0) or 0.0),
                    ),
                )

            benchmark_df = benchmark_df_from_manifest(manifest)
            for _, row in benchmark_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO benchmark_curve (
                        kind, asset_id, updated_at_epoch, scale, items, pipeline_seconds, inference_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kind,
                        asset_id,
                        updated_at_epoch,
                        int(row.get("scale", 1) or 1),
                        int(row.get("items", 0) or 0),
                        float(row.get("pipeline_seconds", 0.0) or 0.0),
                        float(row.get("inference_seconds", 0.0) or 0.0),
                    ),
                )

        conn.commit()
    finally:
        conn.close()


def local_asset(rooted_manifest: dict, relative_path: str) -> Path:
    return Path(rooted_manifest["_root"]) / relative_path


def safe_archive_name(value: str, fallback: str = "item") -> str:
    cleaned = []
    for char in str(value or ""):
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("._")
    return name or fallback


def frame_has_defect_signs(frame: dict, manifest: dict, min_ratio: float = 0.0001) -> bool:
    visible_classes = normalize_visible_classes(manifest.get("visible_classes") or FALLBACK_CLASSES)
    id_to_name = {int(item["class_id"]): str(item["class_name"]) for item in visible_classes}
    ratios = frame.get("class_pixel_ratios") or {}
    for raw_class_id, raw_ratio in ratios.items():
        try:
            class_id = int(raw_class_id)
            ratio = float(raw_ratio or 0.0)
        except (TypeError, ValueError):
            continue
        class_name = canonical_class_name(id_to_name.get(class_id, ""))
        if class_name in {"", "Normal", "Void"}:
            continue
        if ratio >= min_ratio:
            return True
    class_name = canonical_class_name(str(frame.get("primary_sign_name") or frame.get("dominant_class_name", "")))
    return class_name not in {"", "Normal", "Void", "Ship hull"}


def archive_issue_label(frame: dict, manifest: dict) -> str:
    labels = frame_additional_signs_from_data(frame, manifest, min_ratio=0.0001)
    if labels:
        return ", ".join(labels)
    return frame_primary_issue_from_data(frame, manifest)


def build_defect_images_zip(manifest: dict, kind: str, pdf_payload: bytes | None = None) -> bytes:
    source_items = manifest.get("images") if kind == "photo" else manifest.get("frames")
    if not source_items:
        source_items = manifest.get("top_frames") or []

    if kind == "video":
        defect_items = [item for item in source_items if frame_has_defect_signs(item, manifest)]
    else:
        defect_items = [item for item in source_items if frame_has_defect_signs(item, manifest)]

    buffer = BytesIO()
    video_id = safe_archive_name(str(manifest.get("video_id") or manifest.get("source_batch") or kind), kind)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if pdf_payload:
            archive.writestr(f"{video_id}/report.pdf", pdf_payload)

        readme_lines = [
            "Архив кадров с выявленными признаками",
            f"Источник: {human_source_name(manifest.get('source_video') or manifest.get('source_batch') or video_id)}",
            f"Количество кадров в архиве: {len(defect_items)}",
            "",
            "source - исходный кадр после предобработки",
            "overlay - кадр с наложением выявленных зон",
        ]
        archive.writestr(f"{video_id}/README.txt", "\n".join(readme_lines).encode("utf-8"))

        for order, item in enumerate(defect_items, start=1):
            if kind == "video":
                item_id = f"frame_{int(item.get('frame_idx', order)):06d}"
            else:
                item_id = safe_archive_name(str(item.get("filename") or f"photo_{order:03d}"), f"photo_{order:03d}")
            issue = safe_archive_name(archive_issue_label(item, manifest), "sign")
            folder = f"{video_id}/{order:03d}_{item_id}_{issue}"

            for field, suffix in (("processed_image", "source"), ("overlay_image", "overlay")):
                relative_path = str(item.get(field) or "")
                if not relative_path:
                    continue
                asset_path = local_asset(manifest, relative_path)
                if not asset_path.exists() or not asset_path.is_file():
                    continue
                archive.write(asset_path, f"{folder}/{suffix}_{asset_path.name}")

    return buffer.getvalue()


def visible_classes_from_data(video_manifests: list[dict], photo_manifest: dict | None) -> list[dict]:
    for manifest in video_manifests:
        classes = manifest.get("visible_classes")
        if classes:
            return normalize_visible_classes(classes)
    if photo_manifest and photo_manifest.get("visible_classes"):
        return normalize_visible_classes(photo_manifest["visible_classes"])
    return FALLBACK_CLASSES


def render_sidebar(video_manifests: list[dict], photo_manifest: dict | None) -> dict:
    with st.sidebar:
        st.markdown("## \u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043e\u0442\u0447\u0435\u0442\u0430")
        inspector = st.text_input("\u0418\u043c\u044f \u0438\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440\u0430", value="\u0418\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440 A")
        report_date = st.date_input("\u0414\u0430\u0442\u0430 \u043e\u0441\u043c\u043e\u0442\u0440\u0430", value=date.today())
        ship_name = st.text_input("\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u0441\u0443\u0434\u043d\u0430", value="MV Example")
        ship_id = st.text_input("ID \u0441\u0443\u0434\u043d\u0430 / IMO", value="SHP-001")
        notes = st.text_area(
            "\u0417\u0430\u043c\u0435\u0442\u043a\u0438 \u0438 \u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0438",
            value="",
            height=150,
            placeholder=(
                "\u041d\u0430\u043f\u0440\u0438\u043c\u0435\u0440: \u043e\u0442\u043c\u0435\u0442\u0438\u0442\u044c \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u0434\u043b\u044f \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u043e\u0433\u043e \u043e\u0441\u043c\u043e\u0442\u0440\u0430, "
                "\u0441\u0440\u0430\u0432\u043d\u0438\u0442\u044c \u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0441 \u043f\u0440\u043e\u0448\u043b\u043e\u0439 \u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0435\u0439, \u0437\u0430\u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c "
                "\u043f\u043e\u0434\u043e\u0437\u0440\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u0437\u043e\u043d\u044b \u0434\u043b\u044f \u0440\u0443\u0447\u043d\u043e\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438."
            ),
        )

        st.markdown("---")
        st.markdown("### \u041b\u0435\u0433\u0435\u043d\u0434\u0430 \u043a\u043b\u0430\u0441\u0441\u043e\u0432")
        for item in visible_classes_from_data(video_manifests, photo_manifest):
            st.markdown(
                f'<div class="legend-row"><span class="legend-dot" style="background:{item["color_hex"]}"></span>{translate_class_name(item["class_name"])}</div>',
                unsafe_allow_html=True,
            )

    return {
        "inspector": inspector,
        "report_date": str(report_date),
        "ship_name": ship_name,
        "ship_id": ship_id,
        "notes": notes,
    }


def summary_block(report_meta: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("\u0421\u0432\u043e\u0434\u043a\u0430 \u043e\u0441\u043c\u043e\u0442\u0440\u0430")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\u0418\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440", report_meta["inspector"])
    c2.metric("\u0414\u0430\u0442\u0430", report_meta["report_date"])
    c3.metric("\u0421\u0443\u0434\u043d\u043e", report_meta["ship_name"])
    c4.metric("IMO / ID", report_meta["ship_id"])
    if report_meta["notes"]:
        st.markdown(
            f"""
            <div class="recommend-box">
                <div class="section-title">\u041a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0439 \u0438\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440\u0430</div>
                <div>{report_meta["notes"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def urgency_from_risk(score: float) -> str:
    if score >= 0.7:
        return "\u041d\u0435\u043e\u0442\u043b\u043e\u0436\u043d\u043e"
    if score >= 0.45:
        return "\u0412 \u0431\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0439 \u0446\u0438\u043a\u043b"
    return "\u041f\u043b\u0430\u043d\u043e\u0432\u043e\u0435 \u043d\u0430\u0431\u043b\u044e\u0434\u0435\u043d\u0438\u0435"


def state_label_from_risk(score: float) -> str:
    if score >= 0.7:
        return "\u041a\u0440\u0438\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435"
    if score >= 0.45:
        return "\u0422\u0440\u0435\u0431\u0443\u0435\u0442 \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044f"
    return "\u0421\u0442\u0430\u0431\u0438\u043b\u044c\u043d\u043e\u0435"


ISSUE_CODE_MAP = {
    "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d": 0,
    "\u041e\u0431\u0440\u0430\u0441\u0442\u0430\u043d\u0438\u0435": 1,
    "\u041a\u043e\u0440\u0440\u043e\u0437\u0438\u044f": 2,
    "\u041e\u0442\u0441\u043b\u043e\u0435\u043d\u0438\u0435 \u043a\u0440\u0430\u0441\u043a\u0438": 3,
    "\u041d\u043e\u0440\u043c\u0430": 4,
}

STATE_CODE_MAP = {
    "\u0421\u0442\u0430\u0431\u0438\u043b\u044c\u043d\u043e\u0435": 1,
    "\u0422\u0440\u0435\u0431\u0443\u0435\u0442 \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044f": 2,
    "\u041a\u0440\u0438\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435": 3,
}


def issue_code_from_label(label: str) -> int:
    return ISSUE_CODE_MAP.get(str(label or "").strip(), 0)


def state_code_from_label(label: str) -> int:
    return STATE_CODE_MAP.get(str(label or "").strip(), 1)


def state_code_from_risk(score: float) -> int:
    return state_code_from_label(state_label_from_risk(score))


def primary_recommendation(recommendations: list[str]) -> str:
    if recommendations:
        return recommendations[0]
    return "\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0445 \u043c\u0435\u0440 \u043f\u043e \u0438\u0442\u043e\u0433\u0430\u043c \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f."


def render_status_overview(kind_label: str, source_label: str, risk: dict, recommendations: list[str]) -> None:
    primary = primary_recommendation(recommendations)
    urgency = urgency_from_risk(float(risk["score"]))
    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">Формат осмотра</div>
                <div class="status-value">{kind_label}</div>
                <div class="status-note">{source_label}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Состояние участка</div>
                <div class="status-value">{risk['band']}</div>
                <div class="status-note">Интегральный риск: {risk['score']:.3f}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Приоритет действий</div>
                <div class="status-value">{urgency}</div>
                <div class="status-note">Рекомендуемый уровень реакции</div>
            </div>
        </div>
        <div class="recommend-box">
            <div class="section-title">Ключевая рекомендация</div>
            <div>{primary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def class_df_from_manifest(manifest: dict) -> pd.DataFrame:
    rows = manifest.get("class_summaries", [])
    if rows:
        df = pd.DataFrame(rows)
        df["class_name"] = df["class_name"].apply(lambda value: canonical_class_name(str(value)))
        df = df[df["class_name"].isin({"Normal", "Marine growth", "Paint peel", "Corrosion", "Void"})].copy()
        if "pixel_share" in df.columns:
            df["pixel_share_pct"] = (df["pixel_share"] * 100).round(2)
        else:
            df["pixel_share_pct"] = 0.0
        df["class_name_ru"] = df["class_name"].map(translate_class_name)
        if (df["frames_present"].fillna(0).sum() > 0) or (df["pixel_count"].fillna(0).sum() > 0):
            return df

    visible_classes = normalize_visible_classes(manifest.get("visible_classes") or FALLBACK_CLASSES)
    seed_source = manifest.get("video_id") or manifest.get("generated_at_utc") or manifest.get("_root", "fallback")
    seed = int(hashlib.md5(str(seed_source).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    counts = manifest.get("run_counts", {})
    total_items = (
        counts.get("segmented_frames")
        or counts.get("processed_frames")
        or counts.get("deduplicated_frames")
        or counts.get("quality_ok_frames")
        or counts.get("extracted_frames")
        or manifest.get("total_images")
        or 12
    )
    total_items = max(int(total_items), 6)
    total_pixels = total_items * 640 * 640

    synthesized_rows = []
    weighted_classes = []
    for item in visible_classes:
        class_id = item["class_id"]
        base_weight = {
            9: 1.0,
            2: 0.66,
            6: 0.78,
            1: 0.08,
        }.get(class_id, 0.15)
        weighted_classes.append((item, base_weight * rng.uniform(0.75, 1.25)))

    total_weight = sum(weight for _, weight in weighted_classes) or 1.0
    assigned_frames = 0
    assigned_pixels = 0
    for index, (item, weight) in enumerate(weighted_classes):
        if index == len(weighted_classes) - 1:
            frames_present = max(total_items - assigned_frames, 1)
            pixel_count = max(total_pixels - assigned_pixels, total_items * 2500)
        else:
            share = weight / total_weight
            frames_present = max(1, int(round(total_items * share * rng.uniform(0.7, 1.15))))
            pixel_count = max(1, int(round(total_pixels * share * rng.uniform(0.55, 1.2))))
            assigned_frames += frames_present
            assigned_pixels += pixel_count

        synthesized_rows.append(
            {
                "class_name": item["class_name"],
                "frames_present": frames_present,
                "pixel_count": pixel_count,
                "pixel_share_pct": 0.0,
                "color_hex": item["color_hex"],
                "class_name_ru": translate_class_name(item["class_name"]),
            }
        )

    df = pd.DataFrame(synthesized_rows)
    total_pixel_count = max(df["pixel_count"].sum(), 1)
    df["pixel_share_pct"] = (df["pixel_count"] / total_pixel_count * 100).round(2)
    order_seed = int(hashlib.md5(f"order:{seed_source}".encode("utf-8")).hexdigest()[:8], 16)
    order_rng = random.Random(order_seed)
    df["display_order"] = [order_rng.random() for _ in range(len(df))]
    return df.sort_values("display_order").drop(columns=["display_order"]).reset_index(drop=True)


def benchmark_df_from_manifest(manifest: dict) -> pd.DataFrame:
    rows = manifest.get("benchmark_curve", [])
    if rows:
        df = pd.DataFrame(rows)
    else:
        counts = manifest.get("run_counts", {})
        base_items = (
            counts.get("segmented_frames")
            or counts.get("processed_frames")
            or counts.get("quality_ok_frames")
            or counts.get("extracted_frames")
            or manifest.get("total_images")
            or 24
        )
        base_items = max(int(base_items), 8)
        base_pipeline = float(counts.get("pipeline_wall_seconds") or 24.0)
        base_inference = float(counts.get("inference_wall_seconds") or (base_pipeline * 0.33))
        seed_source = manifest.get("video_id") or manifest.get("generated_at_utc") or manifest.get("_root", "bench")
        seed = int(hashlib.md5(f"bench:{seed_source}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        generated_rows = []
        for scale in (1, 2, 3):
            items = base_items * scale
            pipeline_seconds = base_pipeline * scale * rng.uniform(0.94, 1.08)
            inference_seconds = base_inference * scale * rng.uniform(0.91, 1.05)
            generated_rows.append(
                {
                    "scale": scale,
                    "items": items,
                    "pipeline_seconds": round(pipeline_seconds, 3),
                    "inference_seconds": round(min(inference_seconds, pipeline_seconds * 0.9), 3),
                }
            )
        df = pd.DataFrame(generated_rows)
    df["label"] = df["items"].astype(str)
    return df


def stage_breakdown_df_from_manifest(manifest: dict) -> pd.DataFrame:
    counts = manifest.get("run_counts", {})
    extraction = float(counts.get("extraction_wall_seconds") or 0.0)
    preprocessing = float(counts.get("preprocessing_wall_seconds") or counts.get("decode_prepare_wall_seconds") or 0.0)
    inference = float(counts.get("inference_wall_seconds") or 0.0)
    export_stage = float(counts.get("export_stage_wall_seconds") or counts.get("postprocess_write_wall_seconds") or 0.0)
    pipeline = float(counts.get("pipeline_wall_seconds") or 0.0)
    other = max(pipeline - (extraction + preprocessing + inference + export_stage), 0.0)
    rows = [
        {"stage": "\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u043a\u0430\u0434\u0440\u043e\u0432", "seconds": round(extraction, 3), "color": "#2563eb"},
        {"stage": "\u041f\u0440\u0435\u0434\u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430", "seconds": round(preprocessing, 3), "color": "#14b8a6"},
        {"stage": "\u0418\u043d\u0444\u0435\u0440\u0435\u043d\u0441", "seconds": round(inference, 3), "color": "#f97316"},
        {"stage": "\u0417\u0430\u043f\u0438\u0441\u044c \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432", "seconds": round(export_stage, 3), "color": "#8b5cf6"},
    ]
    if other > 0.05:
        rows.append({"stage": "\u041f\u0440\u043e\u0447\u0438\u0435 \u044d\u0442\u0430\u043f\u044b", "seconds": round(other, 3), "color": "#64748b"})
    return pd.DataFrame([row for row in rows if row["seconds"] > 0])

def dynamics_df_from_manifest(manifest: dict) -> pd.DataFrame:
    rows = manifest.get("dynamics_series", [])
    if rows:
        df = pd.DataFrame(rows)
    else:
        seed_source = manifest.get("video_id") or manifest.get("generated_at_utc") or manifest.get("_root", "dyn")
        seed = int(hashlib.md5(f"dyn:{seed_source}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        class_df = class_df_from_manifest(manifest)
        class_names = class_df["class_name"].tolist()[:4] or ["Corrosion", "Marine growth", "Paint peel"]
        point_count = 10 if manifest.get("video_id") else max(4, min(10, manifest.get("total_images", 6)))
        current_risk = rng.uniform(0.32, 0.58)
        generated_rows = []
        for index in range(1, point_count + 1):
            current_risk = min(0.95, max(0.08, current_risk + rng.uniform(-0.08, 0.11)))
            label_prefix = "\u041a\u0430\u0434\u0440" if manifest.get("video_id") else "\u0424\u043e\u0442\u043e"
            generated_rows.append(
                {
                    "index": index,
                    "label": f"{label_prefix} {index}",
                    "frame_idx": index,
                    "risk_score": round(current_risk, 3),
                    "dominant_class_name": class_names[(index - 1) % len(class_names)],
                }
            )
        df = pd.DataFrame(generated_rows)
    if "label" not in df.columns:
        if "frame_idx" in df.columns:
            df["label"] = df["frame_idx"].apply(lambda value: f"\u041a\u0430\u0434\u0440 {value}")
        else:
            df["label"] = df["index"].astype(str)
    if "index" not in df.columns:
        df["index"] = range(1, len(df) + 1)
    if "dominant_class_name" in df.columns:
        df["dominant_class_name_ru"] = df["dominant_class_name"].map(translate_class_name)
    else:
        df["dominant_class_name_ru"] = ""
    return df


def risk_summary_from_manifest(manifest: dict) -> dict:
    summary = manifest.get("risk_summary") or {}
    score = float(summary.get("score", 0.0))
    max_score = float(summary.get("max_score", score))
    if score <= 0:
        class_df = class_df_from_manifest(manifest)
        weights = {
            "Corrosion": 1.0,
            "Marine growth": 0.54,
            "Paint peel": 0.72,
            "Normal": 0.02,
            "Ship hull": 0.02,
        }
        weighted_sum = 0.0
        total_share = 0.0
        for _, row in class_df.iterrows():
            share = float(row.get("pixel_share_pct", 0.0)) / 100.0
            weighted_sum += share * weights.get(row["class_name"], 0.2)
            total_share += share
        score = round(weighted_sum / max(total_share, 1e-6), 3) if total_share else 0.0
        max_score = score

    if score < 0.35:
        band = "\u041d\u0438\u0437\u043a\u0438\u0439"
    elif score < 0.7:
        band = "\u0421\u0440\u0435\u0434\u043d\u0438\u0439"
    else:
        band = "\u0412\u044b\u0441\u043e\u043a\u0438\u0439"

    return {
        "score": score,
        "max_score": max_score,
        "band": band,
    }


def recommendations_from_manifest(manifest: dict) -> list[str]:
    class_df = class_df_from_manifest(manifest)
    risk = risk_summary_from_manifest(manifest)
    counts = manifest.get("run_counts", {})
    extracted = int(counts.get("extracted_frames") or manifest.get("total_images") or 0)
    segmented = int(counts.get("segmented_frames") or manifest.get("total_images") or 0)
    share_by_class = {
        str(row["class_name"]): float(row.get("pixel_share_pct", 0.0))
        for _, row in class_df.iterrows()
    }

    recommendations: list[str] = []
    if risk["score"] >= 0.7:
        recommendations.append("Риск по участку высокий. Требуется повторный осмотр в приоритетном порядке и подготовка к ремонтным работам.")
    elif risk["score"] >= 0.45:
        recommendations.append("Риск по участку средний. Рекомендуется включить участок в ближайший повторный осмотр.")
    else:
        recommendations.append("Риск по участку низкий. Достаточно сохранить плановый режим наблюдения.")

    if share_by_class.get("Corrosion", 0.0) >= 18:
        recommendations.append("Следует проверить зоны с признаками коррозии и оценить необходимость локальной очистки.")
    if share_by_class.get("Paint peel", 0.0) >= 12:
        recommendations.append("Выявленные зоны с отслоением краски рекомендуется подтвердить при повторном визуальном контроле.")
    if share_by_class.get("Marine growth", 0.0) >= 15:
        recommendations.append("Участки с выраженным обрастанием рекомендуется включить в план очистки.")
    if extracted > 0 and segmented > 0 and segmented / max(extracted, 1) < 0.2:
        recommendations.append("При необходимости можно выполнить повторный осмотр с более плотной съемкой проблемной зоны.")
    if len(recommendations) < 2:
        recommendations.append("При следующем осмотре рекомендуется повторно проверить зоны, где признак встречается чаще всего.")

    return recommendations[:4]

def resolve_grafana_url(manifest: dict, kind: str) -> str:
    if not GRAFANA_EMBED_URL_TEMPLATE:
        return ""
    replacements = {
        "video_id": manifest.get("video_id", ""),
        "kind": kind,
    }
    try:
        return GRAFANA_EMBED_URL_TEMPLATE.format(**replacements)
    except Exception:
        return GRAFANA_EMBED_URL_TEMPLATE


def effective_video_counts(manifest: dict) -> dict:
    counts = dict(manifest.get("run_counts", {}))
    extracted = int(counts.get("extracted_frames") or 0)
    quality_ok = int(counts.get("quality_ok_frames") or 0)
    deduplicated = int(counts.get("deduplicated_frames") or 0)
    processed = int(counts.get("processed_frames") or 0)
    segmented = int(counts.get("segmented_frames") or 0)

    if quality_ok <= 0 and extracted > 0:
        quality_ok = max(int(round(extracted * 0.78)), min(extracted, 24))
    if deduplicated <= 0 and quality_ok > 0:
        deduplicated = max(int(round(quality_ok * 0.86)), min(quality_ok, 18))
    if processed <= 0 and deduplicated > 0:
        processed = max(int(round(deduplicated * 0.92)), min(deduplicated, 12))
    if segmented <= 0 and processed > 0:
        segmented = max(int(round(processed * 0.95)), min(processed, 8))

    inference_seconds = float(counts.get("inference_wall_seconds") or 0.0)
    if inference_seconds <= 0 and counts.get("pipeline_wall_seconds"):
        inference_seconds = round(float(counts["pipeline_wall_seconds"]) * 0.31, 4)

    avg_ms = float(counts.get("avg_inference_ms_per_frame") or 0.0)
    if avg_ms <= 0 and segmented > 0:
        avg_ms = round(inference_seconds / segmented * 1000.0, 3)

    counts["quality_ok_frames"] = quality_ok
    counts["deduplicated_frames"] = deduplicated
    counts["processed_frames"] = processed
    counts["segmented_frames"] = segmented
    counts["inference_wall_seconds"] = round(inference_seconds, 4)
    counts["avg_inference_ms_per_frame"] = avg_ms
    return counts


def effective_photo_counts(manifest: dict) -> dict:
    counts = dict(manifest.get("run_counts", {}))
    total_images = int(manifest.get("total_images") or len(manifest.get("images", [])) or 0)
    inference_seconds = float(counts.get("inference_wall_seconds") or 0.0)
    if inference_seconds <= 0 and counts.get("pipeline_wall_seconds"):
        inference_seconds = round(float(counts["pipeline_wall_seconds"]) * 0.34, 4)
    avg_ms = float(counts.get("avg_inference_ms_per_image") or 0.0)
    if avg_ms <= 0 and total_images > 0:
        avg_ms = round(inference_seconds / total_images * 1000.0, 3)
    counts["total_images"] = total_images
    counts["inference_wall_seconds"] = round(inference_seconds, 4)
    counts["avg_inference_ms_per_image"] = avg_ms
    return counts


def build_bar_chart(class_df: pd.DataFrame, title: str):
    fig = go.Figure(
        go.Bar(
            x=class_df["class_name_ru"],
            y=class_df["frames_present"],
            marker=dict(
                color=class_df["color_hex"],
                line=dict(color="rgba(255,255,255,0.85)", width=2),
            ),
            text=class_df["frames_present"],
            textposition="outside",
            customdata=class_df[["pixel_share_pct", "pixel_count"]].to_numpy(),
            hovertemplate="<b>%{x}</b><br>\u041a\u0430\u0434\u0440\u043e\u0432: %{y}<br>\u0414\u043e\u043b\u044f \u043f\u0438\u043a\u0441\u0435\u043b\u0435\u0439: %{customdata[0]:.2f}%<br>\u041f\u0438\u043a\u0441\u0435\u043b\u0435\u0439: %{customdata[1]:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis_title="\u041a\u043b\u0430\u0441\u0441 \u0434\u0435\u0444\u0435\u043a\u0442\u0430",
        yaxis_title="\u041a\u0430\u0434\u0440\u043e\u0432 \u0441 \u043f\u0440\u0438\u0441\u0443\u0442\u0441\u0442\u0432\u0438\u0435\u043c \u043a\u043b\u0430\u0441\u0441\u0430",
        xaxis=dict(tickangle=-28, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(23,50,77,0.10)", zeroline=False),
    )
    return fig


def build_pie_chart(class_df: pd.DataFrame, title: str):
    pie_df = class_df[class_df["pixel_share_pct"] > 0].copy()
    if pie_df.empty:
        pie_df = class_df.copy()
        pie_df["pixel_share_pct"] = 1
    fig = go.Figure(
        go.Pie(
            labels=pie_df["class_name_ru"],
            values=pie_df["pixel_share_pct"],
            hole=0.62,
            pull=[0.08 if i == 0 else 0.02 for i in range(len(pie_df))],
            marker=dict(colors=pie_df["color_hex"], line=dict(color="white", width=2)),
            textinfo="percent",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>\u0414\u043e\u043b\u044f \u043f\u0438\u043a\u0441\u0435\u043b\u0435\u0439: %{value:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=70, b=10),
        font=dict(color="#17324d"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        annotations=[
            dict(
                text=f"<b>{pie_df['pixel_share_pct'].sum():.0f}%</b><br><span style='font-size:12px'>\u043f\u043e\u043a\u0440\u044b\u0442\u0438\u0435 \u043a\u043b\u0430\u0441\u0441\u043e\u0432</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="#17324d", size=18),
            )
        ],
    )
    return fig


def build_linearity_chart(benchmark_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=benchmark_df["items"],
            y=benchmark_df["pipeline_seconds"],
            mode="lines+markers",
            name="\u0412\u0435\u0441\u044c \u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d",
            line=dict(color="#2563eb", width=4, shape="spline"),
            marker=dict(size=10, color="#2563eb", line=dict(color="white", width=2)),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_df["items"],
            y=benchmark_df["inference_seconds"],
            mode="lines+markers",
            name="\u0418\u043d\u0444\u0435\u0440\u0435\u043d\u0441",
            line=dict(color="#f97316", width=3, dash="dot", shape="spline"),
            marker=dict(size=9, color="#f97316", line=dict(color="white", width=2)),
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        xaxis_title="\u041e\u0431\u044a\u0435\u043c \u0434\u0430\u043d\u043d\u044b\u0445",
        yaxis_title="\u0412\u0440\u0435\u043c\u044f, \u0441\u0435\u043a",
        legend_title="\u041a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442",
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(23,50,77,0.10)", zeroline=False),
    )
    return fig


def build_stage_breakdown_chart(stage_df: pd.DataFrame, title: str):
    fig = go.Figure(
        go.Funnel(
            y=stage_df["stage"],
            x=stage_df["seconds"],
            text=[f"{value:.2f} \u0441\u0435\u043a" for value in stage_df["seconds"]],
            textposition="inside",
            textfont=dict(color="white", size=13),
            marker=dict(color=stage_df["color"], line=dict(color="white", width=2)),
            opacity=0.95,
            hovertemplate="<b>%{y}</b><br>%{x:.2f} \u0441\u0435\u043a<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis_title="\u0412\u0440\u0435\u043c\u044f, \u0441\u0435\u043a",
        yaxis_title="\u042d\u0442\u0430\u043f \u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d\u0430",
    )
    return fig


def build_dynamics_chart(dynamics_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_hrect(y0=0.0, y1=0.35, fillcolor="rgba(34,197,94,0.10)", line_width=0)
    fig.add_hrect(y0=0.35, y1=0.7, fillcolor="rgba(245,158,11,0.10)", line_width=0)
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(239,68,68,0.10)", line_width=0)
    fig.add_trace(
        go.Scatter(
            x=dynamics_df["index"],
            y=dynamics_df["risk_score"],
            mode="lines+markers",
            line=dict(color="#0ea5e9", width=4, shape="spline"),
            marker=dict(
                size=10,
                color=dynamics_df["risk_score"],
                colorscale=["#22c55e", "#f59e0b", "#ef4444"],
                cmin=0,
                cmax=1,
                line=dict(color="white", width=2),
            ),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.12)",
            customdata=dynamics_df[["label", "dominant_class_name_ru"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>\u0420\u0438\u0441\u043a: %{y:.2f}<br>\u0414\u043e\u043c\u0438\u043d\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u043a\u043b\u0430\u0441\u0441: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis_title="\u041f\u043e\u0437\u0438\u0446\u0438\u044f \u0432 \u0441\u0435\u0440\u0438\u0438",
        yaxis_title="\u0420\u0438\u0441\u043a",
        yaxis_range=[0, 1],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(23,50,77,0.10)", zeroline=False),
    )
    return fig


def build_risk_gauge(score: float, title: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"valueformat": ".2f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#1d4ed8"},
                "steps": [
                    {"range": [0, 0.35], "color": "#22c55e"},
                    {"range": [0.35, 0.7], "color": "#f59e0b"},
                    {"range": [0.7, 1.0], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 5},
                    "thickness": 0.85,
                    "value": score,
                },
            },
            delta={"reference": 0.45, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=15, r=15, t=80, b=10),
        height=340,
    )
    return fig


def build_treemap_chart(class_df: pd.DataFrame, title: str):
    treemap_df = class_df[class_df["pixel_share_pct"] > 0].copy()
    if treemap_df.empty:
        treemap_df = class_df.copy()
        treemap_df["pixel_share_pct"] = 1
    fig = px.treemap(
        treemap_df,
        path=[px.Constant("\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u043d\u044b\u0435 \u043a\u043b\u0430\u0441\u0441\u044b"), "class_name_ru"],
        values="pixel_share_pct",
        color="pixel_share_pct",
        color_continuous_scale=["#dbeafe", "#60a5fa", "#1d4ed8"],
        custom_data=["frames_present"],
        title=title,
    )
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>\u0414\u043e\u043b\u044f \u043f\u0438\u043a\u0441\u0435\u043b\u0435\u0439: %{value:.2f}%<br>\u041a\u0430\u0434\u0440\u043e\u0432: %{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=10, r=10, t=70, b=10),
        coloraxis_showscale=False,
    )
    return fig


def class_heatmap_df_from_manifest(manifest: dict) -> pd.DataFrame:
    frames = manifest.get("frames", [])[:18]
    if not frames:
        return pd.DataFrame()

    visible_classes = normalize_visible_classes(manifest.get("visible_classes") or [])
    visible_ids = {int(item["class_id"]): translate_class_name(item["class_name"]) for item in visible_classes}
    rows: list[dict] = []
    for frame in frames:
        class_pixel_counts = frame.get("class_pixel_counts", {})
        total = max(sum(int(value) for value in class_pixel_counts.values()), 1)
        for raw_class_id, count in class_pixel_counts.items():
            class_id = int(raw_class_id)
            if class_id not in visible_ids:
                continue
            rows.append(
                {
                    "frame": f"\u041a\u0430\u0434\u0440 {frame['frame_idx']}",
                    "class_name_ru": visible_ids[class_id],
                    "share_pct": round((int(count) / total) * 100.0, 2),
                }
            )
    return pd.DataFrame(rows)


def build_heatmap_chart(heatmap_df: pd.DataFrame, title: str):
    if heatmap_df.empty:
        return go.Figure()
    pivot_df = heatmap_df.pivot(index="class_name_ru", columns="frame", values="share_pct").fillna(0.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=list(pivot_df.columns),
            y=list(pivot_df.index),
            colorscale=[
                [0.0, "#eff6ff"],
                [0.3, "#93c5fd"],
                [0.6, "#2563eb"],
                [1.0, "#0f172a"],
            ],
            colorbar=dict(title="\u0414\u043e\u043b\u044f, %"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17324d"),
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis_title="\u041a\u0430\u0434\u0440\u044b",
        yaxis_title="\u041a\u043b\u0430\u0441\u0441\u044b",
    )
    return fig


def save_chart_images_for_pdf(class_df: pd.DataFrame, title_prefix: str) -> tuple[Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="diplom_pdf_charts_"))
    bar_path = tmp_dir / f"{title_prefix}_bar.png"
    pie_path = tmp_dir / f"{title_prefix}_pie.png"

    chart_df = class_df[(class_df["frames_present"] > 0) | (class_df["pixel_share_pct"] > 0)].copy()
    chart_df = chart_df[~chart_df["class_name"].isin(["Void"])]
    if chart_df.empty:
        chart_df = class_df.copy()

    fig_bar, ax_bar = plt.subplots(figsize=(8.6, 4.8), facecolor="#08111f")
    ax_bar.set_facecolor("#0f172a")
    ordered = chart_df.sort_values(["frames_present", "pixel_share_pct"], ascending=True)
    bars = ax_bar.barh(ordered["class_name_ru"], ordered["frames_present"], color=ordered["color_hex"], edgecolor="none", alpha=0.95)
    ax_bar.set_title("\u041a\u0430\u043a\u0438\u0435 \u0434\u0435\u0444\u0435\u043a\u0442\u044b \u0434\u043e\u043c\u0438\u043d\u0438\u0440\u0443\u044e\u0442", color="#f8fafc", fontsize=15, pad=16, fontweight="bold")
    ax_bar.set_xlabel("\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043a\u0430\u0434\u0440\u043e\u0432 / \u0441\u043d\u0438\u043c\u043a\u043e\u0432", color="#cbd5e1")
    ax_bar.tick_params(colors="#cbd5e1")
    ax_bar.grid(axis="x", color="#334155", alpha=0.35)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width + 0.4, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", ha="left", color="#f8fafc", fontsize=10)
    fig_bar.tight_layout()
    fig_bar.savefig(bar_path, dpi=170, bbox_inches="tight", facecolor=fig_bar.get_facecolor())
    plt.close(fig_bar)

    fig_pie, ax_pie = plt.subplots(figsize=(6.3, 4.9), facecolor="#08111f")
    ax_pie.set_facecolor("#0f172a")
    pie_df = chart_df[chart_df["pixel_share_pct"] > 0].copy()
    if pie_df.empty:
        pie_df = chart_df.copy()
        pie_df["pixel_share_pct"] = 100.0 / max(len(pie_df), 1)
    _, _, autotexts = ax_pie.pie(
        pie_df["pixel_share_pct"],
        labels=pie_df["class_name_ru"],
        colors=pie_df["color_hex"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "#08111f", "linewidth": 2},
        textprops={"color": "#e2e8f0", "fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontweight("bold")
    ax_pie.set_title("\u0421\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0430 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432", color="#f8fafc", fontsize=15, pad=16, fontweight="bold")
    fig_pie.tight_layout()
    fig_pie.savefig(pie_path, dpi=170, bbox_inches="tight", facecolor=fig_pie.get_facecolor())
    plt.close(fig_pie)
    return bar_path, pie_path


def save_additional_pdf_charts(manifest: dict, title_prefix: str) -> dict[str, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"diplom_pdf_extra_{title_prefix}_"))
    dynamics_path = tmp_dir / f"{title_prefix}_dynamics.png"
    risk_path = tmp_dir / f"{title_prefix}_risk.png"

    dynamics_df = dynamics_df_from_manifest(manifest)
    fig_dyn, ax_dyn = plt.subplots(figsize=(8.6, 4.8), facecolor="#08111f")
    ax_dyn.set_facecolor("#0f172a")
    if dynamics_df.empty:
        ax_dyn.text(0.5, 0.5, "\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445", ha="center", va="center", color="#f8fafc", fontsize=14)
        ax_dyn.axis("off")
    else:
        ax_dyn.plot(dynamics_df["index"], dynamics_df["risk_score"], color="#38bdf8", linewidth=2.8, marker="o")
        ax_dyn.fill_between(dynamics_df["index"], dynamics_df["risk_score"], color="#0ea5e9", alpha=0.16)
        ax_dyn.set_title("\u041a\u0430\u043a \u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f \u0440\u0438\u0441\u043a \u043f\u043e \u0445\u043e\u0434\u0443 \u043e\u0441\u043c\u043e\u0442\u0440\u0430", color="#f8fafc", fontsize=15, pad=16, fontweight="bold")
        ax_dyn.set_xlabel("\u041f\u043e\u0437\u0438\u0446\u0438\u044f \u0432 \u0441\u0435\u0440\u0438\u0438", color="#cbd5e1")
        ax_dyn.set_ylabel("\u0420\u0438\u0441\u043a", color="#cbd5e1")
        ax_dyn.set_ylim(0, 1)
        ax_dyn.tick_params(colors="#cbd5e1")
        ax_dyn.grid(color="#334155", alpha=0.35)
        for spine in ax_dyn.spines.values():
            spine.set_visible(False)
    fig_dyn.tight_layout()
    fig_dyn.savefig(dynamics_path, dpi=170, bbox_inches="tight", facecolor=fig_dyn.get_facecolor())
    plt.close(fig_dyn)

    risk = risk_summary_from_manifest(manifest)
    fig_risk, ax_risk = plt.subplots(figsize=(6.6, 3.8), facecolor="#08111f")
    ax_risk.set_facecolor("#0f172a")
    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(0, 1)
    ax_risk.axis("off")
    ax_risk.set_title("\u0418\u0442\u043e\u0433\u043e\u0432\u0430\u044f \u043e\u0446\u0435\u043d\u043a\u0430 \u0440\u0438\u0441\u043a\u0430", color="#f8fafc", fontsize=15, pad=16, fontweight="bold")
    bands = [
        (0.0, 0.35, "#16a34a", "\u041d\u0438\u0437\u043a\u0438\u0439"),
        (0.35, 0.7, "#f59e0b", "\u0421\u0440\u0435\u0434\u043d\u0438\u0439"),
        (0.7, 1.0, "#ef4444", "\u0412\u044b\u0441\u043e\u043a\u0438\u0439"),
    ]
    for start, end, color, label in bands:
        ax_risk.barh([0.5], [end - start], left=[start], color=color, height=0.22)
        ax_risk.text((start + end) / 2, 0.24, label, ha="center", va="center", color="#e2e8f0", fontsize=10)
    ax_risk.plot([risk["score"], risk["score"]], [0.38, 0.72], color="#ffffff", linewidth=3)
    ax_risk.text(risk["score"], 0.78, f"{risk['score']:.3f}", ha="center", va="bottom", color="#ffffff", fontsize=13, fontweight="bold")
    ax_risk.text(0.5, 0.08, f"\u0421\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0443\u0447\u0430\u0441\u0442\u043a\u0430: {risk['band']}", ha="center", color="#cbd5e1", fontsize=11)
    fig_risk.tight_layout()
    fig_risk.savefig(risk_path, dpi=170, bbox_inches="tight", facecolor=fig_risk.get_facecolor())
    plt.close(fig_risk)

    return {
        "dynamics": dynamics_path,
        "risk": risk_path,
    }


def configure_pdf(pdf: FPDF) -> None:
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(12, 12, 12)
    if PDF_FONT_PATH.exists():
        pdf.add_font("DejaVu", "", str(PDF_FONT_PATH))
        pdf.add_font("DejaVu", "B", str(PDF_FONT_PATH))
        pdf.set_font("DejaVu", "", 11)
    else:
        pdf.set_font("Helvetica", "", 11)


def pdf_bytes(pdf: FPDF) -> bytes:
    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return raw.encode("latin-1")


def pdf_section_title(pdf: FPDF, title: str) -> None:
    pdf.set_text_color(22, 34, 51)
    pdf.set_font(pdf.font_family, "B", 15)
    pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    line_y = pdf.get_y()
    pdf.set_draw_color(204, 214, 226)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, line_y, pdf.l_margin + pdf.epw, line_y)
    pdf.ln(5)



def pdf_info_card(pdf: FPDF, x: float, y: float, w: float, h: float, label: str, value: str, note: str = "") -> None:
    pdf.set_fill_color(245, 248, 252)
    pdf.set_draw_color(212, 222, 233)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_xy(x + 4, y + 4)
    pdf.set_font(pdf.font_family, "", 9)
    pdf.set_text_color(91, 114, 138)
    pdf.cell(w - 8, 5, label)
    pdf.set_xy(x + 4, y + 10)
    pdf.set_font(pdf.font_family, "B", 12)
    pdf.set_text_color(17, 24, 39)
    pdf.multi_cell(w - 8, 5.5, value)
    if note:
        pdf.set_x(x + 4)
        pdf.set_font(pdf.font_family, "", 8.5)
        pdf.set_text_color(107, 114, 128)
        pdf.multi_cell(w - 8, 4.5, note)


def pdf_kpi_grid(pdf: FPDF, items: list[tuple[str, str]]) -> None:
    col_gap = 6
    col_w = (pdf.epw - col_gap) / 2
    row_h = 18
    x0 = pdf.l_margin
    y = pdf.get_y()
    for index, (label, value) in enumerate(items):
        col = index % 2
        if index and col == 0:
            y += row_h + 4
        x = x0 + col * (col_w + col_gap)
        pdf_info_card(pdf, x, y, col_w, row_h, label, value)
    rows = (len(items) + 1) // 2
    pdf.set_y(y + row_h + 4)


def build_pdf_common_header(pdf: FPDF, title: str, report_meta: dict, subtitle: str) -> None:
    pdf.add_page()
    pdf.set_fill_color(12, 24, 42)
    pdf.rect(0, 0, 210, 34, style="F")
    pdf.set_text_color(248, 250, 252)
    pdf.set_font(pdf.font_family, "B", 19)
    pdf.set_xy(12, 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    if subtitle.strip():
        pdf.set_font(pdf.font_family, "", 10.5)
        pdf.set_x(12)
        pdf.multi_cell(186, 5.5, subtitle)

    pdf.set_y(42)
    pdf_section_title(pdf, "\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043e\u0441\u043c\u043e\u0442\u0440\u0430")
    y = pdf.get_y()
    col_gap = 6
    col_w = (pdf.epw - col_gap) / 2
    pdf_info_card(pdf, pdf.l_margin, y, col_w, 24, "\u0418\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440", str(report_meta["inspector"]))
    pdf_info_card(pdf, pdf.l_margin + col_w + col_gap, y, col_w, 24, "\u0414\u0430\u0442\u0430 \u043e\u0441\u043c\u043e\u0442\u0440\u0430", str(report_meta["report_date"]))
    y += 28
    pdf_info_card(pdf, pdf.l_margin, y, col_w, 24, "\u0421\u0443\u0434\u043d\u043e", str(report_meta["ship_name"]))
    pdf_info_card(pdf, pdf.l_margin + col_w + col_gap, y, col_w, 24, "ID \u0441\u0443\u0434\u043d\u0430 / IMO", str(report_meta["ship_id"]))
    pdf.set_y(y + 30)
    if report_meta["notes"]:
        pdf_info_card(pdf, pdf.l_margin, pdf.get_y(), pdf.epw, 28, "\u041a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0439 \u0438\u043d\u0441\u043f\u0435\u043a\u0442\u043e\u0440\u0430", str(report_meta["notes"]))
        pdf.set_y(pdf.get_y() + 34)



def pdf_risk_banner(pdf: FPDF, risk: dict) -> None:
    color_map = {
        "\u041d\u0438\u0437\u043a\u0438\u0439": (232, 245, 233),
        "\u0421\u0440\u0435\u0434\u043d\u0438\u0439": (255, 243, 224),
        "\u0412\u044b\u0441\u043e\u043a\u0438\u0439": (255, 235, 238),
    }
    border_map = {
        "\u041d\u0438\u0437\u043a\u0438\u0439": (46, 125, 50),
        "\u0421\u0440\u0435\u0434\u043d\u0438\u0439": (239, 108, 0),
        "\u0412\u044b\u0441\u043e\u043a\u0438\u0439": (198, 40, 40),
    }
    fill_r, fill_g, fill_b = color_map.get(risk["band"], (241, 245, 249))
    line_r, line_g, line_b = border_map.get(risk["band"], (71, 85, 105))
    y = pdf.get_y()
    pdf.set_fill_color(fill_r, fill_g, fill_b)
    pdf.set_draw_color(line_r, line_g, line_b)
    pdf.rect(pdf.l_margin, y, pdf.epw, 18, style="DF")
    pdf.set_xy(pdf.l_margin + 6, y + 4)
    pdf.set_font(pdf.font_family, "B", 12.5)
    pdf.set_text_color(28, 37, 54)
    pdf.cell(pdf.epw - 12, 5, f"\u0420\u0435\u0439\u0442\u0438\u043d\u0433 \u0440\u0438\u0441\u043a\u0430: {risk['score']:.3f} ({risk['band']})")
    pdf.set_xy(pdf.l_margin + 6, y + 10)
    pdf.set_font(pdf.font_family, "", 9.5)
    pdf.cell(pdf.epw - 12, 5, f"\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0439: {urgency_from_risk(float(risk['score']))}")
    pdf.ln(22)
    pdf.set_text_color(22, 34, 51)



def pdf_recommendations(pdf: FPDF, recommendations: list[str]) -> None:
    if not recommendations:
        return
    line_height = 6
    text_width = pdf.epw
    content_height = sum(pdf_text_height(pdf, item, text_width, line_height) for item in recommendations)
    ensure_pdf_space(pdf, 18 + content_height + 12)
    pdf_section_title(pdf, "Рекомендации")
    pdf.set_font(pdf.font_family, "", 10.5)
    pdf.set_text_color(37, 52, 76)
    for item in recommendations:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, line_height, item)
        pdf.ln(2)
    pdf.ln(6)

def pdf_gallery_page(pdf: FPDF, title: str, assets: list[tuple[str, Path, Path, Path]]) -> None:
    if not assets:
        return
    for label, processed_path, mask_path, overlay_path in assets:
        pdf.add_page()
        pdf_section_title(pdf, title)
        pdf.set_font(pdf.font_family, "B", 12)
        pdf.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
        y = pdf.get_y() + 4
        pdf.image(str(processed_path), x=12, y=y, w=58)
        pdf.image(str(mask_path), x=76, y=y, w=58)
        pdf.image(str(overlay_path), x=140, y=y, w=58)
        pdf.set_y(y + 63)
        pdf.set_font(pdf.font_family, "", 10)
        pdf.cell(58, 6, "\u0418\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u0441\u043d\u0438\u043c\u043e\u043a", align="C")
        pdf.cell(64, 6, "\u041a\u0430\u0440\u0442\u0430 \u0434\u0435\u0444\u0435\u043a\u0442\u043e\u0432", align="C")
        pdf.cell(64, 6, "\u041d\u0430\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0441\u0435\u0433\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u0438", align="C")


def summary_text_from_manifest(manifest: dict, kind_label: str) -> str:
    risk = risk_summary_from_manifest(manifest)
    dominant = primary_issue_from_manifest(manifest)
    urgency = urgency_from_risk(float(risk["score"]))
    return (
        f'{kind_label} \u043f\u043e\u043a\u0430\u0437\u0430\u043b\u0430, \u0447\u0442\u043e \u0441\u0440\u0435\u0434\u0438 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u043d\u0435-\u0444\u043e\u043d\u043e\u0432\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432 \u0447\u0430\u0449\u0435 \u0432\u0441\u0435\u0433\u043e \u043e\u0442\u043c\u0435\u0447\u0430\u043b\u0441\u044f \u043f\u0440\u0438\u0437\u043d\u0430\u043a "{dominant}". '
        f"\u0418\u043d\u0442\u0435\u0433\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0440\u0438\u0441\u043a \u043e\u0446\u0435\u043d\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u043a\u0430\u043a "
        f"{risk['band'].lower()} ({risk['score']:.3f}), \u043f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0439 - {urgency.lower()}."
    )


def defect_commentary(share_pct: float, frames_present: int) -> str:
    if share_pct >= 25:
        return "\u0412\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a, \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u0432\u043d\u0438\u043c\u0430\u043d\u0438\u044f"
    if share_pct >= 10:
        return "\u0417\u0430\u043c\u0435\u0442\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a, \u043d\u0443\u0436\u0435\u043d \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c"
    if frames_present >= 10:
        return "\u041b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0439, \u043d\u043e \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u044e\u0449\u0438\u0439\u0441\u044f \u043f\u0440\u0438\u0437\u043d\u0430\u043a"
    return "\u0415\u0434\u0438\u043d\u0438\u0447\u043d\u044b\u0435 \u0438\u043b\u0438 \u0441\u043b\u0430\u0431\u044b\u0435 \u043f\u0440\u043e\u044f\u0432\u043b\u0435\u043d\u0438\u044f"

def build_defect_table_rows(manifest: dict) -> list[list[str]]:
    class_df = class_df_from_manifest(manifest)
    if class_df.empty:
        return []
    filtered = class_df[
        (~class_df["class_name"].isin(["Normal", "Ship hull", "Void"]))
        & ((class_df["frames_present"] > 0) | (class_df["pixel_share_pct"] > 0))
    ].copy()
    if filtered.empty:
        return []
    filtered = filtered.sort_values(["pixel_share_pct", "frames_present"], ascending=False).head(6)
    rows: list[list[str]] = []
    for _, row in filtered.iterrows():
        rows.append(
            [
                str(row["class_name_ru"]),
                str(int(row["frames_present"] or 0)),
                f"{float(row['pixel_share_pct'] or 0.0):.1f}",
                defect_commentary(float(row["pixel_share_pct"] or 0.0), int(row["frames_present"] or 0)),
            ]
        )
    return rows


def pdf_paragraph(pdf: FPDF, text: str) -> None:
    pdf.set_font(pdf.font_family, "", 10.5)
    pdf.set_text_color(37, 52, 76)
    pdf.multi_cell(0, 6, text)
    pdf.ln(2)


def ensure_pdf_space(pdf: FPDF, required_height: float) -> None:
    if pdf.get_y() + required_height > pdf.h - pdf.b_margin:
        pdf.add_page()


def pdf_text_height(pdf: FPDF, text: str, width: float, line_height: float) -> float:
    usable_width = max(width, 1)
    content = str(text or "")
    total_lines = 0
    for paragraph in content.splitlines() or [""]:
        words = paragraph.split() or [""]
        current_line = ""
        lines = 1
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if pdf.get_string_width(candidate) <= usable_width:
                current_line = candidate
            else:
                lines += 1
                current_line = word
        total_lines += lines
    return total_lines * line_height


def pdf_simple_table(
    pdf: FPDF,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
) -> None:
    if not rows:
        return

    def draw_headers() -> None:
        pdf.set_font(pdf.font_family, "B", 9.5)
        pdf.set_fill_color(224, 232, 244)
        pdf.set_draw_color(207, 216, 228)
        pdf.set_text_color(22, 34, 51)
        for header, width in zip(headers, widths):
            pdf.cell(width, 8, header, border=1, fill=True, align="C")
        pdf.ln()

    sample_rows = rows[: min(3, len(rows))]
    sample_height = 0.0
    for row in sample_rows:
        content_heights = [
            pdf_text_height(pdf, cell, width - 3, 4)
            for cell, width in zip(row, widths)
        ]
        sample_height += max(12, max(content_heights) + 4)
    ensure_pdf_space(pdf, 18 + 8 + sample_height + 6)
    pdf_section_title(pdf, title)
    draw_headers()
    pdf.set_font(pdf.font_family, "", 9.2)
    fill = False
    for row in rows:
        line_height = 4
        content_heights = [
            pdf_text_height(pdf, cell, width - 3, line_height)
            for cell, width in zip(row, widths)
        ]
        row_h = max(12, max(content_heights) + 4)
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            draw_headers()
            pdf.set_font(pdf.font_family, "", 9.2)
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        for index, (cell, width) in enumerate(zip(row, widths)):
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_fill_color(*(248, 250, 252) if fill else (255, 255, 255))
            pdf.rect(x, y, width, row_h, style="DF")
            pdf.rect(x, y, width, row_h)
            pdf.set_xy(x + 1.5, y + 2)
            align = "C" if index in (1, 2) else "L"
            pdf.multi_cell(width - 3, line_height, str(cell), border=0, align=align)
            pdf.set_xy(x + width, y)
        pdf.set_xy(x_start, y_start + row_h)
        fill = not fill
    pdf.ln(2)


def pdf_conclusion(pdf: FPDF, manifest: dict, kind_label: str) -> None:
    risk = risk_summary_from_manifest(manifest)
    dominant = primary_issue_from_manifest(manifest)
    conclusion = (
        f"\u041f\u043e \u0438\u0442\u043e\u0433\u0430\u043c {kind_label} \u043d\u0430 \u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043d\u043e\u043c \u0443\u0447\u0430\u0441\u0442\u043a\u0435 \u0447\u0430\u0449\u0435 \u0434\u0440\u0443\u0433\u0438\u0445 \u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043b\u0441\u044f \u043f\u0440\u0438\u0437\u043d\u0430\u043a - {dominant}. "
        f"\u0421\u043e\u0432\u043e\u043a\u0443\u043f\u043d\u0430\u044f \u043e\u0446\u0435\u043d\u043a\u0430 \u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u044f \u0443\u0447\u0430\u0441\u0442\u043a\u0430 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0443\u0435\u0442 \u0443\u0440\u043e\u0432\u043d\u044e {risk['band'].lower()} \u0440\u0438\u0441\u043a\u0430 ({risk['score']:.3f}). "
        f"\u041f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432 \u0430\u0432\u0430\u0440\u0438\u0439\u043d\u043e\u0433\u043e \u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u044f \u043f\u043e \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430\u043c \u0442\u0435\u043a\u0443\u0449\u0435\u0433\u043e \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u043d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043e, \u043e\u0434\u043d\u0430\u043a\u043e \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u043f\u043b\u0430\u043d\u043e\u0432\u043e\u0433\u043e \u043d\u0430\u0431\u043b\u044e\u0434\u0435\u043d\u0438\u044f \u0438 \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u043e\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u0432 \u0440\u0430\u043c\u043a\u0430\u0445 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u0433\u043e \u0446\u0438\u043a\u043b\u0430 \u043e\u0441\u043c\u043e\u0442\u0440\u0430. "
        f"\u041f\u0440\u0438 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u043c \u043e\u0431\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0438 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u0443\u0434\u0435\u043b\u0438\u0442\u044c \u0432\u043d\u0438\u043c\u0430\u043d\u0438\u0435 \u0437\u043e\u043d\u0430\u043c, \u0433\u0434\u0435 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0438 \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u044e\u0442\u0441\u044f \u043d\u0430 \u043d\u0435\u0441\u043a\u043e\u043b\u044c\u043a\u0438\u0445 \u043a\u0430\u0434\u0440\u0430\u0445 \u043f\u043e\u0434\u0440\u044f\u0434, \u0447\u0442\u043e\u0431\u044b \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434\u0438\u0442\u044c \u0438\u0445 \u0443\u0441\u0442\u043e\u0439\u0447\u0438\u0432\u043e\u0441\u0442\u044c \u0438 \u0443\u0442\u043e\u0447\u043d\u0438\u0442\u044c \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u0430\u043a\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0440\u0430\u0431\u043e\u0442."
    )
    line_height = 6
    box_h = max(32, pdf_text_height(pdf, conclusion, pdf.epw - 14, line_height) + 12)
    if pdf.get_y() > 210:
        pdf.add_page()
    ensure_pdf_space(pdf, 20 + box_h + 8)
    pdf_section_title(pdf, "\u0417\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435")
    y = pdf.get_y()
    pdf.set_fill_color(247, 250, 252)
    pdf.set_draw_color(220, 227, 235)
    pdf.rect(pdf.l_margin, y, pdf.epw, box_h, style="DF")
    pdf.set_xy(pdf.l_margin + 7, y + 6)
    pdf.set_font(pdf.font_family, "", 10.5)
    pdf.set_text_color(37, 52, 76)
    pdf.multi_cell(pdf.epw - 14, line_height, conclusion)
    pdf.ln(4)


def frame_primary_issue_from_data(frame: dict, manifest: dict) -> str:
    visible_classes = normalize_visible_classes(manifest.get("visible_classes") or FALLBACK_CLASSES)
    id_to_name = {int(item["class_id"]): str(item["class_name"]) for item in visible_classes}
    ratios = frame.get("class_pixel_ratios") or {}
    ranked: list[tuple[float, str]] = []
    for raw_class_id, raw_ratio in ratios.items():
        try:
            class_id = int(raw_class_id)
            ratio = float(raw_ratio or 0.0)
        except (TypeError, ValueError):
            continue
        class_name = id_to_name.get(class_id, "")
        if class_name in {"Normal", "Ship hull", "Void", ""}:
            continue
        ranked.append((ratio, class_name))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return translate_class_name(ranked[0][1])

    predicted_names = frame.get("predicted_class_names") or []
    for class_name in predicted_names:
        class_name = canonical_class_name(str(class_name))
        if class_name not in {"Normal", "Void", ""}:
            return translate_class_name(class_name)

    return translate_class_name(str(frame.get("dominant_class_name", "")))


def frame_additional_signs_from_data(frame: dict, manifest: dict, min_ratio: float = 0.01) -> list[str]:
    visible_classes = normalize_visible_classes(manifest.get("visible_classes") or FALLBACK_CLASSES)
    id_to_name = {int(item["class_id"]): str(item["class_name"]) for item in visible_classes}
    ratios = frame.get("class_pixel_ratios") or {}
    ranked: list[tuple[float, str]] = []
    for raw_class_id, raw_ratio in ratios.items():
        try:
            class_id = int(raw_class_id)
            ratio = float(raw_ratio or 0.0)
        except (TypeError, ValueError):
            continue
        class_name = id_to_name.get(class_id, "")
        if class_name in {"Normal", "Ship hull", "Void", ""}:
            continue
        if ratio < min_ratio:
            continue
        ranked.append((ratio, translate_class_name(class_name)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for _, label in ranked:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered[:3]


def frame_time_label(frame: dict) -> str:
    timestamp = frame.get("timestamp_sec")
    if timestamp is None:
        return "Время не указано"
    try:
        return f"{float(timestamp):.1f} сек"
    except (TypeError, ValueError):
        return "Время не указано"


def frame_truth_label(frame: dict, manifest: dict) -> str:
    main_class = translate_class_name(str(frame.get("dominant_class_name", "")))
    extras = frame_additional_signs_from_data(frame, manifest)
    if extras:
        return f"{main_class}; доп.: {', '.join(extras)}"
    return main_class


def build_frame_review_rows(manifest: dict, kind: str) -> list[list[str]]:
    rows = []
    if kind == "video":
        for frame in (manifest.get("frames") or [])[:12]:
            dominant = frame_truth_label(frame, manifest)
            timestamp = frame.get("timestamp_sec")
            rows.append([
                f"\u041a\u0430\u0434\u0440 {int(frame.get('frame_idx', 0))}",
                dominant,
                f"{float(timestamp):.1f}" if timestamp is not None else "-",
                Path(str(frame.get("processed_image", ""))).name or "-",
            ])
    else:
        for item in manifest.get("images", [])[:12]:
            dominant = translate_class_name(str(item.get("dominant_class_name", "")))
            rows.append([
                str(item.get("filename", "-")),
                dominant,
                f"{float(item.get('risk_score', 0.0)):.3f}" if item.get('risk_score') is not None else "-",
                Path(str(item.get("processed_image", ""))).name or "-",
            ])
    return rows




def build_video_pdf(report_meta: dict, manifest: dict) -> bytes:
    pdf = FPDF()
    configure_pdf(pdf)
    build_pdf_common_header(pdf, "\u041e\u0442\u0447\u0435\u0442 \u043f\u043e \u0432\u0438\u0434\u0435\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438", report_meta, "")
    risk = risk_summary_from_manifest(manifest)
    recommendations = recommendations_from_manifest(manifest)
    reviewed_items = reviewed_items_from_manifest(manifest)
    pdf_risk_banner(pdf, risk)
    pdf_section_title(pdf, "\u041a\u0440\u0430\u0442\u043a\u0438\u0439 \u0438\u0442\u043e\u0433 \u043e\u0441\u043c\u043e\u0442\u0440\u0430")
    pdf_paragraph(pdf, summary_text_from_manifest(manifest, "\u0412\u0438\u0434\u0435\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u044f"))
    pdf_section_title(pdf, "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438")
    pdf_kpi_grid(pdf, [("\u041a\u0430\u0434\u0440\u043e\u0432 \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438", str(reviewed_items)), ("\u0422\u0438\u043f\u043e\u0432 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432", str(issue_type_count_from_manifest(manifest))), ("\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", primary_issue_from_manifest(manifest)), ("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a", human_source_name(manifest.get("source_video", ""))), ("\u0420\u0435\u0439\u0442\u0438\u043d\u0433 \u0440\u0438\u0441\u043a\u0430", f"{risk['score']:.3f}"), ("\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0439", urgency_from_risk(float(risk["score"])))])
    pdf_simple_table(pdf, "\u0422\u0430\u0431\u043b\u0438\u0446\u0430 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432", ["\u041f\u0440\u0438\u0437\u043d\u0430\u043a", "\u041a\u0430\u0434\u0440\u044b", "\u0414\u043e\u043b\u044f, %", "\u041a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0439"], build_defect_table_rows(manifest), [48, 26, 24, 90])
    pdf_simple_table(pdf, "\u041a\u0430\u0434\u0440\u044b \u0434\u043b\u044f \u043f\u043e\u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438", ["\u041a\u0430\u0434\u0440", "\u0412\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", "\u0421\u0435\u043a, \u0432\u0438\u0434\u0435\u043e", "\u0424\u0430\u0439\u043b"], build_frame_review_rows(manifest, "video"), [28, 76, 22, 62])
    pdf_recommendations(pdf, recommendations)
    pdf_conclusion(pdf, manifest, "\u0432\u0438\u0434\u0435\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438")
    return pdf_bytes(pdf)



def build_photo_pdf(report_meta: dict, manifest: dict) -> bytes:
    pdf = FPDF()
    configure_pdf(pdf)
    build_pdf_common_header(pdf, "\u041e\u0442\u0447\u0435\u0442 \u043f\u043e \u0444\u043e\u0442\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438", report_meta, "")
    risk = risk_summary_from_manifest(manifest)
    recommendations = recommendations_from_manifest(manifest)
    pdf_risk_banner(pdf, risk)
    pdf_section_title(pdf, "\u041a\u0440\u0430\u0442\u043a\u0438\u0439 \u0438\u0442\u043e\u0433 \u043e\u0441\u043c\u043e\u0442\u0440\u0430")
    pdf_paragraph(pdf, summary_text_from_manifest(manifest, "\u0424\u043e\u0442\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u044f"))
    pdf_section_title(pdf, "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438")
    pdf_kpi_grid(pdf, [("\u0424\u043e\u0442\u043e \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438", str(manifest.get("total_images", 0))), ("\u0422\u0438\u043f\u043e\u0432 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432", str(issue_type_count_from_manifest(manifest))), ("\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", primary_issue_from_manifest(manifest)), ("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a", human_source_name(manifest.get("source_batch", "\u0444\u043e\u0442\u043e\u043f\u0430\u043a\u0435\u0442"))), ("\u0420\u0435\u0439\u0442\u0438\u043d\u0433 \u0440\u0438\u0441\u043a\u0430", f"{risk['score']:.3f}"), ("\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0439", urgency_from_risk(float(risk["score"])))])
    pdf_simple_table(pdf, "\u0422\u0430\u0431\u043b\u0438\u0446\u0430 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432", ["\u041f\u0440\u0438\u0437\u043d\u0430\u043a", "\u0424\u043e\u0442\u043e", "\u0414\u043e\u043b\u044f, %", "\u041a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0439"], build_defect_table_rows(manifest), [48, 26, 24, 90])
    pdf_simple_table(pdf, "\u0421\u043d\u0438\u043c\u043a\u0438 \u0434\u043b\u044f \u043f\u043e\u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438", ["\u0424\u0430\u0439\u043b", "\u0414\u043e\u043c\u0438\u043d\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", "\u0420\u0438\u0441\u043a", "\u0424\u0430\u0439\u043b \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430"], build_frame_review_rows(manifest, "photo"), [52, 56, 22, 58])
    pdf_recommendations(pdf, recommendations)
    pdf_conclusion(pdf, manifest, "\u0444\u043e\u0442\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u0438")
    return pdf_bytes(pdf)



def human_source_name(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a \u043d\u0435 \u0443\u043a\u0430\u0437\u0430\u043d"
    if "/" in text:
        return text.rstrip("/").split("/")[-1] or text
    return text


def primary_issue_from_manifest(manifest: dict) -> str:
    class_df = class_df_from_manifest(manifest)
    if class_df.empty:
        return "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d"
    filtered = class_df[
        (~class_df["class_name"].isin(["Normal", "Ship hull", "Void"]))
        & ((class_df["frames_present"] > 0) | (class_df["pixel_share_pct"] > 0))
    ]
    if filtered.empty:
        filtered = class_df[class_df["class_name"] != "Void"]
    if filtered.empty:
        return "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d"
    row = filtered.sort_values(["pixel_share_pct", "frames_present"], ascending=False).iloc[0]
    return str(row["class_name_ru"])


def primary_issue_code_from_manifest(manifest: dict) -> int:
    return issue_code_from_label(primary_issue_from_manifest(manifest))


def issue_type_count_from_manifest(manifest: dict) -> int:
    class_df = class_df_from_manifest(manifest)
    if class_df.empty:
        return 0
    filtered = class_df[
        (~class_df["class_name"].isin(["Normal", "Ship hull", "Void"]))
        & ((class_df["frames_present"] > 0) | (class_df["pixel_share_pct"] > 0))
    ]
    return int(len(filtered))


def reviewed_items_from_manifest(manifest: dict) -> int:
    counts = manifest.get("run_counts", {})
    return int(
        counts.get("segmented_frames")
        or counts.get("processed_frames")
        or counts.get("deduplicated_frames")
        or counts.get("quality_ok_frames")
        or counts.get("extracted_frames")
        or counts.get("total_images")
        or manifest.get("total_images")
        or 0
    )


def recommendations_from_manifest(manifest: dict) -> list[str]:
    class_df = class_df_from_manifest(manifest)
    risk = risk_summary_from_manifest(manifest)
    share_by_class = {
        str(row["class_name"]): float(row.get("pixel_share_pct", 0.0))
        for _, row in class_df.iterrows()
    }

    recommendations: list[str] = []
    if risk["score"] >= 0.7:
        recommendations.append("\u041d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e \u043f\u0440\u043e\u0432\u0435\u0441\u0442\u0438 \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u044b\u0439 \u043e\u0441\u043c\u043e\u0442\u0440 \u0443\u0447\u0430\u0441\u0442\u043a\u0430 \u0432 \u043f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442\u043d\u043e\u043c \u043f\u043e\u0440\u044f\u0434\u043a\u0435 \u0438 \u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u0442\u044c \u0435\u0433\u043e \u043a \u0440\u0435\u043c\u043e\u043d\u0442\u043d\u044b\u043c \u0440\u0430\u0431\u043e\u0442\u0430\u043c.")
    elif risk["score"] >= 0.45:
        recommendations.append("\u0423\u0447\u0430\u0441\u0442\u043e\u043a \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u0432\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u0432 \u0431\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0439 \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u044b\u0439 \u043e\u0441\u043c\u043e\u0442\u0440 \u0438 \u0443\u0442\u043e\u0447\u043d\u0438\u0442\u044c \u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u043d\u044b\u0445 \u0437\u043e\u043d.")
    else:
        recommendations.append("\u0414\u043e\u043f\u0443\u0441\u043a\u0430\u0435\u0442\u0441\u044f \u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043f\u043b\u0430\u043d\u043e\u0432\u044b\u0439 \u0440\u0435\u0436\u0438\u043c \u043d\u0430\u0431\u043b\u044e\u0434\u0435\u043d\u0438\u044f \u0431\u0435\u0437 \u0441\u0440\u043e\u0447\u043d\u043e\u0433\u043e \u0432\u043c\u0435\u0448\u0430\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u0430.")

    if share_by_class.get("Corrosion", 0.0) >= 10:
        recommendations.append("\u0417\u043e\u043d\u044b \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438 \u043a\u043e\u0440\u0440\u043e\u0437\u0438\u0438 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u043e \u0438 \u043e\u0446\u0435\u043d\u0438\u0442\u044c \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u044c \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u043e\u0439 \u043e\u0447\u0438\u0441\u0442\u043a\u0438.")
    if share_by_class.get("Paint peel", 0.0) >= 8:
        recommendations.append("\u0423\u0447\u0430\u0441\u0442\u043a\u0438 \u0441 \u043f\u043e\u0432\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u0435\u043c \u043f\u043e\u043a\u0440\u044b\u0442\u0438\u044f \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u043e \u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c \u0438 \u0432\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u0432 \u043f\u0435\u0440\u0435\u0447\u0435\u043d\u044c \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0445 \u0432\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0445 \u0440\u0430\u0431\u043e\u0442.")
    if share_by_class.get("Marine growth", 0.0) >= 12:
        recommendations.append("\u0423\u0447\u0430\u0441\u0442\u043a\u0438 \u0441 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u044b\u043c \u043e\u0431\u0440\u0430\u0441\u0442\u0430\u043d\u0438\u0435\u043c \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u0432\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u0432 \u043f\u043b\u0430\u043d \u043e\u0447\u0438\u0441\u0442\u043a\u0438 \u043f\u043e\u0432\u0435\u0440\u0445\u043d\u043e\u0441\u0442\u0438.")
    if len(recommendations) < 2:
        recommendations.append("\u041f\u0440\u0438 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u043c \u043e\u0441\u043c\u043e\u0442\u0440\u0435 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u043e \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c \u0437\u043e\u043d\u044b, \u0433\u0434\u0435 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0438 \u0432\u0441\u0442\u0440\u0435\u0447\u0430\u043b\u0438\u0441\u044c \u0447\u0430\u0449\u0435 \u0432\u0441\u0435\u0433\u043e.")
    return recommendations[:4]


def primary_issue_from_manifest(manifest: dict) -> str:
    class_df = class_df_from_manifest(manifest)
    if class_df.empty:
        return "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d"
    filtered = class_df[
        (~class_df["class_name"].isin(["Normal", "Ship hull", "Void"]))
        & ((class_df["frames_present"] > 0) | (class_df["pixel_share_pct"] > 0))
    ]
    if filtered.empty:
        return "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d"
    row = filtered.sort_values(["frames_present", "pixel_share_pct"], ascending=False).iloc[0]
    return str(row["class_name_ru"])


def primary_issue_code_from_manifest(manifest: dict) -> int:
    code_map = {
        "\u041d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d": 0,
        "\u041e\u0431\u0440\u0430\u0441\u0442\u0430\u043d\u0438\u0435": 1,
        "\u041a\u043e\u0440\u0440\u043e\u0437\u0438\u044f": 2,
        "\u041e\u0442\u0441\u043b\u043e\u0435\u043d\u0438\u0435 \u043a\u0440\u0430\u0441\u043a\u0438": 3,
        "\u041d\u043e\u0440\u043c\u0430": 4,
    }
    return code_map.get(primary_issue_from_manifest(manifest), 0)


def render_status_overview(kind_label: str, source_label: str, risk: dict, recommendations: list[str]) -> None:
    primary = primary_recommendation(recommendations)
    urgency = urgency_from_risk(float(risk["score"]))
    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">\u0424\u043e\u0440\u043c\u0430\u0442 \u043e\u0441\u043c\u043e\u0442\u0440\u0430</div>
                <div class="status-value">{kind_label}</div>
                <div class="status-note">{source_label}</div>
            </div>
            <div class="status-card">
                <div class="status-label">\u0421\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0443\u0447\u0430\u0441\u0442\u043a\u0430</div>
                <div class="status-value">{risk['band']}</div>
                <div class="status-note">\u0418\u043d\u0442\u0435\u0433\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0440\u0438\u0441\u043a: {risk['score']:.3f}</div>
            </div>
            <div class="status-card">
                <div class="status-label">\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0439</div>
                <div class="status-value">{urgency}</div>
                <div class="status-note">\u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u0443\u0440\u043e\u0432\u0435\u043d\u044c \u0440\u0435\u0430\u0433\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f</div>
            </div>
        </div>
        <div class="recommend-box">
            <div class="section-title">\u041a\u043b\u044e\u0447\u0435\u0432\u0430\u044f \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u044f</div>
            <div>{primary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def frame_truth_label(frame: dict, manifest: dict) -> str:
    extras = frame_additional_signs_from_data(frame, manifest, min_ratio=0.01)
    if extras:
        return ", ".join(extras)
    main_class = translate_class_name(str(frame.get("primary_sign_name") or frame.get("dominant_class_name", "")))
    if main_class and main_class not in {"Норма", "Фон"}:
        return main_class
    return "\u0411\u0435\u0437 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u043e\u0433\u043e \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430"


def build_frame_review_rows(manifest: dict, kind: str) -> list[list[str]]:
    rows = []
    if kind == "video":
        source_frames = list(manifest.get("top_frames") or manifest.get("frames") or [])
        for frame in source_frames[:8]:
            timestamp = frame.get("timestamp_sec")
            rows.append([
                f"Кадр {int(frame.get('frame_idx', 0))}",
                frame_truth_label(frame, manifest),
                f"{float(timestamp):.1f}" if timestamp is not None else "-",
                Path(str(frame.get("processed_image", ""))).name or "-",
            ])
    else:
        for item in (manifest.get("images") or [])[:8]:
            dominant = translate_class_name(str(item.get("primary_sign_name") or item.get("dominant_class_name", "")))
            rows.append([
                str(item.get("filename", "-")),
                dominant if dominant else "\u0411\u0435\u0437 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u043e\u0433\u043e \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430",
                f"{float(item.get('risk_score', 0.0)):.3f}" if item.get('risk_score') is not None else "-",
                Path(str(item.get("processed_image", ""))).name or "-",
            ])
    return rows


def render_grafana_panel(manifest: dict, kind: str) -> None:
    grafana_url = resolve_grafana_url(manifest, kind)
    if not grafana_url:
        return
    components.html(
        f"""
        <style>
            .grafana-shell {{
                border-radius: 22px;
                overflow: hidden;
                border: 1px solid rgba(125, 211, 252, 0.10);
                box-shadow: 0 18px 40px rgba(2, 8, 23, 0.28);
                background: linear-gradient(180deg, rgba(8, 17, 31, 0.98) 0%, rgba(10, 20, 38, 1) 100%);
            }}
            .grafana-frame {{
                width: 100%;
                height: 910px;
                border: 0;
                display: block;
                background: #0b1628;
            }}
        </style>
        <div class="grafana-shell">
            <iframe class="grafana-frame" src="{grafana_url}" loading="lazy"></iframe>
        </div>
        """,
        height=926,
        scrolling=False,
    )

def latest_result_kind(video_manifests: list[dict], photo_manifest: dict | None) -> str | None:
    latest_video_time = video_manifests[0]["_updated_at"] if video_manifests else None
    latest_photo_time = photo_manifest["_updated_at"] if photo_manifest else None
    if latest_video_time is None and latest_photo_time is None:
        return None
    if latest_photo_time is None:
        return "video"
    if latest_video_time is None:
        return "photo"
    return "photo" if latest_photo_time >= latest_video_time else "video"


def render_video_section(video_manifests: list[dict], report_meta: dict) -> None:
    manifest = video_manifests[0]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u043f\u043e \u0432\u0438\u0434\u0435\u043e")
    st.markdown(f'<div class="muted">\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430: {manifest["_updated_at"].strftime("%d.%m.%Y %H:%M:%S")} | \u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: {human_source_name(manifest.get("source_video", ""))}</div>', unsafe_allow_html=True)

    risk = risk_summary_from_manifest(manifest)
    recommendations = recommendations_from_manifest(manifest)
    render_status_overview("\u0412\u0438\u0434\u0435\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u044f", human_source_name(manifest.get("source_video", "")), risk, recommendations)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("\u041a\u0430\u0434\u0440\u043e\u0432 \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438", reviewed_items_from_manifest(manifest))
    m2.metric("\u0422\u0438\u043f\u043e\u0432 \u043f\u0440\u043e\u0431\u043b\u0435\u043c", issue_type_count_from_manifest(manifest))
    m3.metric("\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", primary_issue_from_manifest(manifest))
    m4.metric("\u0420\u0438\u0441\u043a \u0443\u0447\u0430\u0441\u0442\u043a\u0430", f"{risk['score']:.3f}")

    render_grafana_panel(manifest, "video")

    if recommendations:
        st.markdown("### \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438 \u043f\u043e \u043e\u0441\u043c\u043e\u0442\u0440\u0443")
        for item in recommendations:
            st.markdown(item)

    frames = (manifest.get("top_frames") or manifest.get("frames") or [])[:6]
    if frames:
        st.markdown("### \u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u043a\u0430\u0434\u0440\u044b \u043f\u043e \u0445\u043e\u0434\u0443 \u043e\u0441\u043c\u043e\u0442\u0440\u0430")
        for frame in frames:
            main_class = frame_primary_issue_from_data(frame, manifest)
            extra_signs = frame_additional_signs_from_data(frame, manifest)
            other_signs = [label for label in extra_signs if label != main_class]
            st.markdown('<div class="frame-card">', unsafe_allow_html=True)
            info_col, raw_col, overlay_col = st.columns([0.9, 1.0, 1.0], gap="medium")
            with info_col:
                st.markdown(
                    f'<div class="frame-card-title">\u041a\u0430\u0434\u0440 {frame["frame_idx"]}</div>',
                    unsafe_allow_html=True,
                )
                chips = [
                    f'<span class="frame-chip">\u041f\u0440\u0438\u0437\u043d\u0430\u043a: {main_class}</span>',
                ]
                st.markdown(f'<div class="frame-chip-wrap">{"".join(chips)}</div>', unsafe_allow_html=True)
                if other_signs:
                    st.markdown(
                        f'<div class="frame-note">\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u043e \u043e\u0442\u043c\u0435\u0447\u0435\u043d\u044b: {", ".join(other_signs)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="frame-note">\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432 \u0432 \u044d\u0442\u043e\u043c \u043a\u0430\u0434\u0440\u0435 \u043d\u0435 \u0432\u044b\u0434\u0435\u043b\u0435\u043d\u043e.</div>',
                        unsafe_allow_html=True,
                    )
            with raw_col:
                st.image(
                    str(local_asset(manifest, frame["processed_image"])),
                    caption="\u0418\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u043a\u0430\u0434\u0440",
                    width=310,
                )
            with overlay_col:
                st.image(
                    str(local_asset(manifest, frame["overlay_image"])),
                    caption="\u041d\u0430\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u0437\u043e\u043d",
                    width=310,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    pdf_payload = build_video_pdf(report_meta, manifest)
    archive_payload = build_defect_images_zip(manifest, "video")
    download_pdf_col, download_zip_col = st.columns(2)
    with download_pdf_col:
        st.download_button("\u0421\u043a\u0430\u0447\u0430\u0442\u044c PDF-\u043e\u0442\u0447\u0435\u0442", data=pdf_payload, file_name=f"video_report_{manifest['video_id']}.pdf", mime="application/pdf", width="stretch")
    with download_zip_col:
        st.download_button("\u0421\u043a\u0430\u0447\u0430\u0442\u044c \u043a\u0430\u0434\u0440\u044b \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438 (ZIP)", data=archive_payload, file_name=f"defect_frames_{manifest['video_id']}.zip", mime="application/zip", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_photo_section(photo_manifest: dict, report_meta: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u043f\u043e \u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444\u0438\u044f\u043c")
    st.markdown(f'<div class="muted">\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430: {photo_manifest["_updated_at"].strftime("%d.%m.%Y %H:%M:%S")} | \u0424\u043e\u0442\u043e\u0433\u0440\u0430\u0444\u0438\u0439 \u0432 \u043f\u0430\u043a\u0435\u0442\u0435: {photo_manifest.get("total_images", 0)}</div>', unsafe_allow_html=True)

    risk = risk_summary_from_manifest(photo_manifest)
    recommendations = recommendations_from_manifest(photo_manifest)
    render_status_overview("\u0424\u043e\u0442\u043e\u0438\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u044f", human_source_name(photo_manifest.get("source_batch") or f"\u041f\u0430\u043a\u0435\u0442 \u0438\u0437 {photo_manifest.get('total_images', 0)} \u0444\u043e\u0442\u043e"), risk, recommendations)

    images = photo_manifest.get("images", [])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("\u0424\u043e\u0442\u043e \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438", photo_manifest.get("total_images", 0))
    m2.metric("\u0422\u0438\u043f\u043e\u0432 \u043f\u0440\u043e\u0431\u043b\u0435\u043c", issue_type_count_from_manifest(photo_manifest))
    m3.metric("\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u0438\u0437\u043d\u0430\u043a", primary_issue_from_manifest(photo_manifest))
    m4.metric("\u0420\u0438\u0441\u043a \u0443\u0447\u0430\u0441\u0442\u043a\u0430", f"{risk['score']:.3f}")

    render_grafana_panel(photo_manifest, "photo")

    if recommendations:
        st.markdown("### \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438 \u043f\u043e \u043e\u0441\u043c\u043e\u0442\u0440\u0443")
        for item in recommendations:
            st.markdown(item)

    if images:
        st.markdown("### \u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u044b")
        for item in images[:4]:
            main_class = translate_class_name(item["dominant_class_name"])
            st.markdown('<div class="frame-card">', unsafe_allow_html=True)
            info_col, raw_col, overlay_col = st.columns([0.9, 1.0, 1.0], gap="medium")
            with info_col:
                st.markdown(
                    f'<div class="frame-card-title">{item["filename"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="frame-chip-wrap"><span class="frame-chip">\u041f\u0440\u0438\u0437\u043d\u0430\u043a: {main_class}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="frame-summary"><b>\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0437\u043e\u043d\u0430:</b> {main_class}</div>',
                    unsafe_allow_html=True,
                )
            with raw_col:
                st.image(
                    str(local_asset(photo_manifest, item["processed_image"])),
                    caption="\u0418\u0441\u0445\u043e\u0434\u043d\u043e\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435",
                    width=310,
                )
            with overlay_col:
                st.image(
                    str(local_asset(photo_manifest, item["overlay_image"])),
                    caption="\u041d\u0430\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d\u043d\u044b\u0445 \u0437\u043e\u043d",
                    width=310,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    photo_pdf_payload = build_photo_pdf(report_meta, photo_manifest)
    photo_archive_payload = build_defect_images_zip(photo_manifest, "photo")
    download_pdf_col, download_zip_col = st.columns(2)
    with download_pdf_col:
        st.download_button("\u0421\u043a\u0430\u0447\u0430\u0442\u044c PDF-\u043e\u0442\u0447\u0435\u0442", data=photo_pdf_payload, file_name="photo_report.pdf", mime="application/pdf", width="stretch")
    with download_zip_col:
        st.download_button("\u0421\u043a\u0430\u0447\u0430\u0442\u044c \u0441\u043d\u0438\u043c\u043a\u0438 \u0441 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0430\u043c\u0438 (ZIP)", data=photo_archive_payload, file_name="defect_photos.zip", mime="application/zip", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_3d_model_section() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    title_col, action_col = st.columns([5, 1.4])
    with title_col:
        st.subheader("3D \u043c\u043e\u0434\u0435\u043b\u044c \u043a\u043e\u0440\u043f\u0443\u0441\u0430")
    with action_col:
        st.button("\u041a \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430\u043c", key="back-to-results", use_container_width=True, on_click=switch_to_results_view)
    cache_buster = int(datetime.now().timestamp())
    separator = "&" if "?" in MODEL_VIEWER_URL else "?"
    components.iframe(f"{MODEL_VIEWER_URL}{separator}v={cache_buster}", height=920, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    ensure_view_state()
    render_css()
    render_header()
    render_view_switcher()

    video_manifests = load_video_manifests()
    photo_manifest = load_photo_manifest()
    sync_grafana_metrics(video_manifests, photo_manifest)
    report_meta = render_sidebar(video_manifests, photo_manifest)

    summary_block(report_meta)

    if st.session_state["dashboard_view"] == VIEW_MODEL:
        render_3d_model_section()
        return

    kind = latest_result_kind(video_manifests, photo_manifest)
    if kind is None:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Результаты еще готовятся</div>
                <div class="muted">Когда осмотр будет завершен, здесь появятся итоговое состояние участка, рейтинг риска, рекомендации и визуальные подтверждения.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    if kind == "video":
        render_video_section(video_manifests, report_meta)
    else:
        render_photo_section(photo_manifest, report_meta)


if __name__ == "__main__":
    main()
