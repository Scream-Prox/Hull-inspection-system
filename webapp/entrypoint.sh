#!/bin/sh
set -e

MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-8502}"
MODEL_DIST_DIR="${MODEL_DIST_DIR:-/opt/3d-model}"
STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-8501}"
STREAMLIT_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

python -m http.server "${MODEL_SERVER_PORT}" --directory "${MODEL_DIST_DIR}" &

exec streamlit run app.py --server.port="${STREAMLIT_PORT}" --server.address="${STREAMLIT_ADDRESS}"
