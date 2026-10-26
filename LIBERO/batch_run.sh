#!/bin/bash
# Thin wrapper — delegates to batch_run/batch_run.py
# Usage:
#   bash batch_run.sh              # dev mode (mission_ids)
#   bash batch_run.sh --mode test  # test mode (tested_mission_ids)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python batch_run/batch_run.py "$@"