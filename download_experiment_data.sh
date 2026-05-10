#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-./data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

$PYTHON_BIN -m pip install -r requirements.txt

# Warm the default MS MARCO document dataset used by the configs.
$PYTHON_BIN scripts/download_experiment_data.py --root "$ROOT_DIR" --download-msmarco

# Download the default BEIR datasets used by the transfer config.
$PYTHON_BIN scripts/download_experiment_data.py --root "$ROOT_DIR" --download-beir --beir-datasets nfcorpus scifact fiqa

echo "All requested datasets have been prepared or queued in cache."
