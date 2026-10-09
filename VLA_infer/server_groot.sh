#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Uniform policy server launcher — GROOT N1.6.
#
# Usage:
#     bash server_groot.sh run_profile/ZeroShot/smoke_groot16.yaml
#     bash server_groot.sh run_profile/ZeroShot/full_groot16.yaml
#
# Or via env var:
#     RUNFILE=run_profile/ZeroShot/smoke_groot16.yaml bash server_groot.sh
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE="/data3/zhixu/LIBERO-PeRM/VLA_infer/run_profile/ZeroShot/smoke_groot16.yaml"

# ── pick runfile (positional arg, else $RUNFILE, else default) ──
if [[ "${1:-}" && "${1:0:2}" != "--" ]]; then
    RUNFILE="$1"
    shift
else
    RUNFILE="${RUNFILE:-$DEFAULT_RUNFILE}"
fi
if [[ ! -f "${RUNFILE}" && -f "${SCRIPT_DIR}/${RUNFILE}" ]]; then
    RUNFILE="${SCRIPT_DIR}/${RUNFILE}"
fi
if [[ ! -f "${RUNFILE}" ]]; then
    echo "Runfile not found: ${RUNFILE}" >&2
    exit 1
fi
RUNFILE="$(readlink -f "${RUNFILE}")"

# ── parse server: section + validate vla field ──
PARSED="$(python3 - "${RUNFILE}" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
s = cfg.get("server") or {}
vla = s.get("vla")
if vla != "groot":
    sys.exit(f"vla mismatch: yaml says {vla!r}, server_groot.sh serves 'groot'")
for k in ("name", "config", "checkpoint", "port"):
    if not s.get(k):
        sys.exit(f"server.{k} missing in runfile")
def q(v): return shlex.quote(str(v) if v is not None else "")
print(f"NAME={q(s['name'])}")
print(f"CONFIG={q(s['config'])}")
print(f"CKPT={q(s['checkpoint'])}")
print(f"PORT={q(s['port'])}")
print(f"CUDA={q(s.get('cuda_devices') or '')}")
print(f"DEVICE={q(s.get('device') or 'cuda:0')}")
PY
)" || { echo "Failed to parse runfile" >&2; exit 1; }
eval "${PARSED}"

if [[ -n "${CUDA}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA}"
fi

# ── announce ──
echo "──────────────────────────────────────────────────────────────"
echo "  Policy server — GROOT N1.6"
echo "──────────────────────────────────────────────────────────────"
echo "  Runfile       : ${RUNFILE}"
echo "  Name          : ${NAME}"
echo "  Config        : ${CONFIG}"
echo "  Checkpoint    : ${CKPT}"
echo "  Port          : ${PORT}"
echo "  CUDA_VISIBLE  : ${CUDA:-(inherit)}"
echo "──────────────────────────────────────────────────────────────"

# ── run ──
# Bypass `uv run` (which auto-syncs from groot_repo's pyproject.toml and
# may downgrade transformers, breaking AutoModel registration for Gr00tN1d6).
# Use the pre-built venv's python directly with PYTHONPATH set so
# `import gr00t.model` resolves to the repo and registers Gr00tN1d6.
GROOT_VENV=/data1/vla-reasoning/virtual_envs/groot
# `groot_repo` was upgraded to N1.7; for N1.6 ckpts we need `groot_repo_n1d6`
# which still registers Gr00tN1d6 with AutoConfig/AutoModel. Pick by config name.
GROOT_REPO=/data3/zhixu/LIBERO-PeRM/VLAs/groot_repo
case "${CONFIG}" in
  *n16*|*N16*|*n1d6*|*N1d6*) GROOT_REPO=/data3/zhixu/LIBERO-PeRM/VLAs/groot_repo_n1d6 ;;
esac
export PYTHONPATH="${GROOT_REPO}:${PYTHONPATH:-}"
export GROOT_REPO   # serve.py reads this to sys.path-insert the right repo
cd "$GROOT_REPO"
exec "$GROOT_VENV/bin/python" /data3/zhixu/LIBERO-PeRM/VLAs/groot/serve.py "${RUNFILE}" "$@"
