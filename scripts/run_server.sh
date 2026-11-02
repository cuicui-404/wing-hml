#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
METHODS="${METHODS:-hml ds np ds_maml np_maml}"
SEEDS="${SEEDS:-42 43 44 45 46}"
: "${DATA_ROOT:?Set DATA_ROOT to the server source-data directory}"
: "${TARGET_CSV:?Set TARGET_CSV to the server 64-condition CSV}"

# Intentional word splitting: methods and seeds are space-separated name lists.
exec "$PYTHON" scripts/run_server.py --config "$CONFIG" --data-root "$DATA_ROOT" --target "$TARGET_CSV" --methods $METHODS --seeds $SEEDS
