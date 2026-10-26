#!/usr/bin/env bash
# Uniform policy server launcher for OpenVLA-OFT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE="${SCRIPT_DIR}/run_profile/ZeroShot/OpenVLA_OFT/openvla_oft_4in1_singpref_0shot.yaml"

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

PARSED="$(python - "${RUNFILE}" <<'PY'
import shlex
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
s = cfg.get("server") or {}
if s.get("vla") != "openvla_oft":
    sys.exit(f"vla mismatch: yaml says {s.get('vla')!r}, server_openvla_oft.sh serves 'openvla_oft'")
for key in ("name", "config", "checkpoint", "port"):
    if not s.get(key):
        sys.exit(f"server.{key} missing in runfile")

def q(value):
    return shlex.quote(str(value) if value is not None else "")

print(f"NAME={q(s['name'])}")
print(f"CONFIG={q(s['config'])}")
print(f"CKPT={q(s['checkpoint'])}")
print(f"PORT={q(s['port'])}")
print(f"CUDA={q(s.get('cuda_devices') or '')}")
PY
)" || {
    echo "Failed to parse runfile" >&2
    exit 1
}
eval "${PARSED}"

if [[ -n "${CUDA}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA}"
fi
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "──────────────────────────────────────────────────────────────"
echo "  Policy server — OpenVLA-OFT"
echo "──────────────────────────────────────────────────────────────"
echo "  Runfile       : ${RUNFILE}"
echo "  Name          : ${NAME}"
echo "  Config        : ${CONFIG}"
echo "  Checkpoint    : ${CKPT}"
echo "  Port          : ${PORT}"
echo "  CUDA_VISIBLE  : ${CUDA:-(inherit)}"
echo "  HF_HOME       : ${HF_HOME}"
echo "──────────────────────────────────────────────────────────────"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/VLAs/openvla_oft"
exec python serve.py "${RUNFILE}" "$@"
