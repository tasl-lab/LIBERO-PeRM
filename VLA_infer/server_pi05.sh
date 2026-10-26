#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Uniform policy server launcher — pi0.5 (openpi).
#
# Lives in VLAs/ alongside run_client.sh. Takes the SAME runfile yaml
# that the client uses — the yaml's server: block carries both the
# server launch info (config, checkpoint, port, CUDA devices) and the
# identity info the client validates against.
#
# Usage:
#     bash server_pi05.sh run_profile/ZeroShot/smoke_pi05.yaml
#     bash server_pi05.sh run_profile/ZeroShot/full_pi05.yaml
#     bash server_pi05.sh run_profile/ZeroShot/full_pi05_p8001.yaml
#
# Or via env var:
#     RUNFILE=run_profile/ZeroShot/smoke_pi05.yaml bash server_pi05.sh
#
# Pass-through extra args go to openpi/serve.py.
# ────────────────────────────────────────────────────────────────────


 set -euo pipefail

# `uv` lives in ~/.local/bin which is not on PATH inside conda envs
# (libero_DataGen / base both work now).
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE="/data3/zhixu/LIBERO-PeRM/VLA_infer/run_profile/ZeroShot/D_pi05_official.yaml"

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
    echo "❌ Runfile not found: ${RUNFILE}" >&2
    exit 1
fi
# Absolutize — we `cd` into openpi/ before invoking serve.py
RUNFILE="$(readlink -f "${RUNFILE}")"

# ── parse server: section + validate vla field ──
PARSED="$(python - "${RUNFILE}" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
s = cfg.get("server") or {}
vla = s.get("vla")
if vla not in ("pi05", "pi0"):
    sys.exit(f"vla mismatch: yaml says {vla!r}, server_pi05.sh serves 'pi05' / 'pi0' (openpi shared)")
for k in ("name", "config", "checkpoint", "port"):
    if not s.get(k):
        sys.exit(f"server.{k} missing in runfile")
def q(v): return shlex.quote(str(v) if v is not None else "")
print(f"NAME={q(s['name'])}")
print(f"CONFIG={q(s['config'])}")
print(f"CKPT={q(s['checkpoint'])}")
print(f"PORT={q(s['port'])}")
print(f"CUDA={q(s.get('cuda_devices') or '')}")
PY
)" || { echo "❌ Failed to parse runfile" >&2; exit 1; }
eval "${PARSED}"

if [[ -n "${CUDA}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA}"
fi

# ── announce (symmetric with run_client.sh) ──
echo "──────────────────────────────────────────────────────────────"
echo "  Policy server — pi0.5 (openpi)"
echo "──────────────────────────────────────────────────────────────"
echo "  Runfile       : ${RUNFILE}"
echo "  Name          : ${NAME}"
echo "  Config        : ${CONFIG}"
echo "  Checkpoint    : ${CKPT}"
echo "  Port          : ${PORT}"
echo "  CUDA_VISIBLE  : ${CUDA:-(inherit)}"
echo "──────────────────────────────────────────────────────────────"

# ── run ──
cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
exec uv run serve.py "${RUNFILE}" "$@"
