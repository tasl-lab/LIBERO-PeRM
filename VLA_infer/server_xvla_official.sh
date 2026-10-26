#!/usr/bin/env bash
# Uniform policy server launcher — X-VLA via official deploy.py (FastAPI HTTP).
# Reads runfile yaml, extracts LoRA checkpoint + port + cuda_devices,
# launches deploy.py from xvla_official_repo. Pair with PolicyEval/eval.py
# (HttpBackend) on the client side.

set -euo pipefail

XVLA_REPO=/data3/zhixu/LIBERO-PeRM/VLAs/xvla_official_repo
XVLA_VENV=/data1/vla-reasoning/virtual_envs/xvla_official
CKPT_BASE=/data3/zhixu/LIBERO-PeRM/checkpoints/xvla_official

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE="$SCRIPT_DIR/run_profile/ZeroShot/v2/full_xvla_object_v2.yaml"

if [[ "${1:-}" && "${1:0:2}" != "--" ]]; then
    RUNFILE="$1"; shift
else
    RUNFILE="${RUNFILE:-$DEFAULT_RUNFILE}"
fi
[[ -f "$RUNFILE" ]] || { echo "❌ Runfile not found: $RUNFILE" >&2; exit 1; }
RUNFILE="$(readlink -f "$RUNFILE")"

# Parse runfile
PARSED="$(python - "$RUNFILE" <<'PY'
import shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
s = cfg.get("server") or {}
if s.get("vla") != "xvla":
    sys.exit(f"vla mismatch: yaml says {s.get('vla')!r}, server_xvla_official.sh serves 'xvla'")
for k in ("name", "port", "checkpoint"):
    if not s.get(k):
        sys.exit(f"server.{k} missing")
def q(v): return shlex.quote(str(v) if v is not None else "")
print(f"NAME={q(s['name'])}")
print(f"PORT={q(s['port'])}")
print(f"LORA={q(s['checkpoint'])}")    # runfile's `checkpoint` field is the LoRA dir
print(f"CUDA={q(s.get('cuda_devices') or '')}")
PY
)" || { echo "❌ Failed to parse runfile" >&2; exit 1; }
eval "$PARSED"

[[ -n "$CUDA" ]] && export CUDA_VISIBLE_DEVICES="$CUDA"
OUT_INFO="${LOGS:-/tmp}/xvla_${NAME}_info"
rm -rf "$OUT_INFO"; mkdir -p "$OUT_INFO"

echo "──────────────────────────────────────────────────────────────"
echo "  Policy server — X-VLA (official deploy.py + FastAPI HTTP)"
echo "──────────────────────────────────────────────────────────────"
echo "  Runfile : $RUNFILE"
echo "  Name    : $NAME"
echo "  LoRA    : $LORA"
echo "  Port    : $PORT"
echo "  CUDA    : ${CUDA:-(inherit)}"
echo "──────────────────────────────────────────────────────────────"

cd "$XVLA_REPO"
exec env PYTHONPATH="$XVLA_REPO" "$XVLA_VENV/bin/python" deploy.py \
    --model_path "$CKPT_BASE/X-VLA-Pt" \
    --LoRA_path "$LORA" \
    --output_dir "$OUT_INFO" \
    --port "$PORT" \
    --disable_slurm
