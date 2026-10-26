#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mission_id> [-- extra args passed to main.py]" >&2
  exit 2
fi

MISSION_ID="$1"
shift

PAIRING_KEY="${MISSION_ID%%_*}"
LIBERO_NUM="${MISSION_ID#*_}"
PERM_NAME="perm_${PAIRING_KEY//-/_}"
LIBERO_NAME="libero_${LIBERO_NUM}"
MISSION_DIR="$(ls -d "$SCRIPT_DIR"/dataset/series_*/"$PERM_NAME"/"$LIBERO_NAME" 2>/dev/null | head -n 1)"

if [[ -z "$MISSION_DIR" || ! -d "$MISSION_DIR" ]]; then
  echo "Mission directory not found: $MISSION_DIR" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

DEFAULT_MAIN_ARGS=(
  --naive_llm_plan 0
  --run_layout_gen 0
  --high_pref_gpt_plan 0
  --low_pref_gpt_plan 0
  --generate_evaluation_data 0
  --enable_collision_check 1
  --batch_size 2
  --batch_start 0
  --manually_set_offset 1
  --check_init_layout 0
  --gen_in_batch 1
  --overwrite_old 1
  --use_forcibly 1
)

cd "$SCRIPT_DIR"

CMD=(
  "$PYTHON_BIN"
  main.py
  --mission_id "$MISSION_ID"
  "${DEFAULT_MAIN_ARGS[@]}"
  "$@"
)

printf 'Running mission: %s\n' "$MISSION_ID"
printf 'Using Python: %s\n' "$PYTHON_BIN"
printf 'Mission dir: %s\n' "$MISSION_DIR"
printf '$'
printf ' %q' "${CMD[@]}"
printf '\n\n'

exec "${CMD[@]}"
