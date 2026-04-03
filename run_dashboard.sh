#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export XDG_CACHE_HOME="$PWD/.streamlit_cache"
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_TELEMETRY_ENABLED=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

source .venv/bin/activate

PORT="${1:-8503}"
streamlit run app_pyless.py --server.port "$PORT" --server.address 127.0.0.1 --server.headless true

