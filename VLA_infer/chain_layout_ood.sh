#!/usr/bin/env bash
# Chain runner for 7 PrefScale layout-OOD eval cells.
# Pairs run on GPU 4 + GPU 7 simultaneously.
# Usage: bash chain_layout_ood.sh
set -uo pipefail

INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer
PROFILE=$INFER/run_profile/LoRA/Pi05
TS=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/pscale_layout_ood_${TS}
mkdir -p "$LOG"
echo "Logs → $LOG"

run_pair() {
    local C1=$1 C2=${2:-}
    echo "[$(date +%H:%M:%S)] ── Pair: ${C1}${C2:+ + $C2} ──"

    # Start servers (background)
    bash $INFER/server_pi05.sh $PROFILE/H_${C1}_layout_ood.yaml > "$LOG/${C1}_server.log" 2>&1 &
    SP1=$!
    if [[ -n "$C2" ]]; then
        bash $INFER/server_pi05.sh $PROFILE/H_${C2}_layout_ood.yaml > "$LOG/${C2}_server.log" 2>&1 &
        SP2=$!
    fi

    echo "  servers up (PIDs: $SP1${SP2:+ $SP2}). Waiting 90s for JIT…"
    sleep 90

    # Start clients (background)
    bash $INFER/run_client.sh $PROFILE/H_${C1}_layout_ood.yaml > "$LOG/${C1}_client.log" 2>&1 &
    CP1=$!
    if [[ -n "$C2" ]]; then
        bash $INFER/run_client.sh $PROFILE/H_${C2}_layout_ood.yaml > "$LOG/${C2}_client.log" 2>&1 &
        CP2=$!
        wait $CP1 $CP2
    else
        wait $CP1
    fi

    echo "[$(date +%H:%M:%S)] clients done. Killing servers."
    kill $SP1 ${SP2:-} 2>/dev/null || true
    pkill -f "serve.py.*H_${C1}_layout_ood" 2>/dev/null || true
    [[ -n "$C2" ]] && pkill -f "serve.py.*H_${C2}_layout_ood" 2>/dev/null || true
    sleep 8
}

run_pair M1 M2
run_pair M3 M4
run_pair L1 L3
run_pair L4

echo "[$(date +%H:%M:%S)] ── ALL DONE ── (log: $LOG)"
