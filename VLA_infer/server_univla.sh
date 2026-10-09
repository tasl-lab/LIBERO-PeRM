#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Uniform policy server launcher — UniVLA.
#
# Loads qwbu/univla-7b-224-sft-libero (an OpenVLA-based VLA + a small
# action decoder head + LAM-generated latent-action vocabulary).
# Pipeline:
#   image + task → VLA predicts latent-action tokens (+ visual embed)
#   latent + visual → ActionDecoderHead → (chunk, 7) normalized action
#   unnormalize (per-suite q01/q99) + binarize+flip gripper
#
# Usage:
#     bash server_univla.sh run_profile/ZeroShot/smoke_univla.yaml
#     bash server_univla.sh run_profile/ZeroShot/full_univla.yaml
#
# Or via env var:
#     RUNFILE=run_profile/ZeroShot/smoke_univla.yaml bash server_univla.sh
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE="/home/vla-reasoning/proj/P_Bench/VLA_infer/run_profile/ZeroShot/smoke_univla.yaml"

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
RUNFILE="$(readlink -f "${RUNFILE}")"

# ── parse server: section + validate vla field ──
PARSED="$(python - "${RUNFILE}" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
s = cfg.get("server") or {}
if s.get("vla") != "univla":
    sys.exit(f"vla mismatch: yaml says {s.get('vla')!r}, server_univla.sh serves 'univla'")
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

echo "──────────────────────────────────────────────────────────────"
echo "  Policy server — UniVLA"
echo "──────────────────────────────────────────────────────────────"
echo "  Runfile       : ${RUNFILE}"
echo "  Name          : ${NAME}"
echo "  Config        : ${CONFIG}"
echo "  Checkpoint    : ${CKPT}"
echo "  Port          : ${PORT}"
echo "  CUDA_VISIBLE  : ${CUDA:-(inherit)}"
echo "──────────────────────────────────────────────────────────────"

cd /data3/zhixu/LIBERO-PeRM/VLAs/univla
exec /data1/vla-reasoning/virtual_envs/univla/bin/python serve.py "${RUNFILE}" "$@"
