#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# P_Bench PolicyEval rollout client launcher.
#
# Lives in VLAs/ (run/inference side). The actual rollout code lives in
# P_Bench/LIBERO/PolicyEval/eval.py (data/env side); this script just
# activates the right conda env, sets PYTHONPATH so `libero.libero`
# resolves to the P_Bench fork, and shells into eval.py.
#
# Usage:
#     bash run_client.sh run_profile/ZeroShot/openvla-7b-oft-finetuned-libero-spatial-object-goal-10_smoke.yaml
#     bash run_client.sh run_profile/ZeroShot/openvla-7b-oft-finetuned-libero-spatial-object-goal-10_full.yaml
#     bash run_client.sh run_profile/ZeroShot/openvla-7b-oft-finetuned-libero-spatial-object-goal-10_smoke.yaml --dry_run
#
# Or via env var:
#     RUNFILE=run_profile/ZeroShot/openvla-7b-oft-finetuned-libero-spatial-object-goal-10_full.yaml bash run_client.sh
#
# Pass-through extra args go to PolicyEval/eval.py.
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-libero_DataGen}"
LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/LIBERO}"
DEFAULT_RUNFILE="${REPO_ROOT}/VLA_infer/run_profile/ZeroShot/OpenVLA_OFT/openvla_oft_4in1_singpref_0shot.yaml"

# ── pick runfile (positional arg, else $RUNFILE, else default) ──
if [[ "${1:-}" && "${1:0:2}" != "--" ]]; then
    RUNFILE="$1"
    shift
else
    RUNFILE="${RUNFILE:-$DEFAULT_RUNFILE}"
fi
if [[ ! -f "${RUNFILE}" ]]; then
    # allow either absolute or relative-to-SCRIPT_DIR (where run_profile/ lives)
    if [[ -f "${SCRIPT_DIR}/${RUNFILE}" ]]; then
        RUNFILE="${SCRIPT_DIR}/${RUNFILE}"
    else
        echo "❌ Runfile not found: ${RUNFILE}" >&2
        exit 1
    fi
fi
# Absolutize before `cd` into LIBERO_ROOT so Python can still open the same file.
RUNFILE="$(readlink -f "${RUNFILE}")"

# ── activate conda env ──
# shellcheck disable=SC1091
CONDA_SH="${CONDA_SH:-/home/zli538/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# ── cd into LIBERO_ROOT + PYTHONPATH so `libero.libero` resolves to P_Bench fork ──
cd "${LIBERO_ROOT}"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"

# ── announce ──
echo "──────────────────────────────────────────────────────────────"
echo "  P_Bench PolicyEval rollout client"
echo "──────────────────────────────────────────────────────────────"
echo "  conda env     : ${CONDA_ENV}"
echo "  Script dir    : ${SCRIPT_DIR}   (VLAs/P_bench)"
echo "  LIBERO source : ${LIBERO_ROOT}/libero   (P_Bench fork)"
echo "  NUMBA cache   : ${NUMBA_CACHE_DIR}"
python - <<'PY'
import os, libero.libero as _m
print(f"  Resolved from : {os.path.dirname(_m.__file__)}")
PY
echo "  Runfile       : ${RUNFILE}"
python - "${RUNFILE}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
srv = cfg.get("server", {}) or {}
print(f"  Experiment    : {cfg.get('experiment_name')}")
print(f"  Exp type      : {cfg.get('exp_type')}")
print(f"  Task suite    : {cfg.get('task_suite')}")
ms = cfg.get("missions") or []
print(f"  Missions      : {len(ms) if ms else '(all of suite)'}"
      + (f"  e.g. {ms[:3]}" if ms else ""))
print(f"  Conditions    : {cfg.get('conditions', ['base', 'pref'])}")
print(f"  Num trials    : {cfg.get('num_trials', 5)}")
print(f"  Server        : ws://{srv.get('host', '0.0.0.0')}:{srv.get('port', '???')}  "
      f"[vla={srv.get('vla', '?')} name={srv.get('name', '?')}]")
print(f"  Expect config : {srv.get('config')!r}")
print(f"  Expect ckpt   : {srv.get('checkpoint')!r}")
PY
echo "──────────────────────────────────────────────────────────────"

# ── run ──
python PolicyEval/eval.py "${RUNFILE}" "$@"
EVAL_RC=$?

# ── post-eval: SR + SA aggregation → evalresult.xlsx ──
if [[ "${EVAL_RC}" -eq 0 ]]; then
    echo "──────────────────────────────────────────────────────────────"
    echo "Post-eval: aggregating SR + SA into evalresult.xlsx"
    python tools/eval_summary.py --runfile "${RUNFILE}" || echo "[eval_summary] failed (non-fatal)"
fi
exit "${EVAL_RC}"
