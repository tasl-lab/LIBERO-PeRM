#!/usr/bin/env bash
# 4-GPU pref-OOD chain. Phase 1: M1/M2/M3/M4 on GPU 0/1/2/3 (4 parallel).
# Phase 2: L1/L3/L4 on GPU 0/1/2 (3 parallel).
# Each cell × 4 pref serial; new server per (cell, pref) due to __end__ bug.
set -uo pipefail

INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer
PROFILE=$INFER/run_profile/LoRA/Pi05
LOG=${LOG_DIR:-/tmp/pscale_pref_ood_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG"
echo "Logs → $LOG"

run_iter_set() {
    # args: list of cells; runs all in parallel for one pref P=$1, then waits.
    local P=$1; shift
    local cells=("$@")
    local SPIDS=() CPIDS=()
    echo "[$(date +%H:%M:%S)]   pref=$P  cells: ${cells[*]}"
    # start servers
    for C in "${cells[@]}"; do
        bash $INFER/server_pi05.sh $PROFILE/H_${C}_pref_ood_${P}.yaml > "$LOG/${C}_${P}_server.log" 2>&1 &
        SPIDS+=($!)
    done
    sleep 90
    # start clients
    for C in "${cells[@]}"; do
        bash $INFER/run_client.sh $PROFILE/H_${C}_pref_ood_${P}.yaml > "$LOG/${C}_${P}_client.log" 2>&1 &
        CPIDS+=($!)
    done
    wait "${CPIDS[@]}"
    # kill servers
    for pid in "${SPIDS[@]}"; do kill $pid 2>/dev/null || true; done
    for C in "${cells[@]}"; do pkill -f "serve.py.*H_${C}_pref_ood_${P}" 2>/dev/null || true; done
    sleep 6
}

run_phase() {
    local PHASE_NAME=$1; shift
    local cells=("$@")
    echo "[$(date +%H:%M:%S)] ── ${PHASE_NAME}: ${cells[*]} ──"
    for P in speed left red multi; do
        run_iter_set "$P" "${cells[@]}"
    done
}

run_phase "Phase 1" M1 M2 M3 M4
run_phase "Phase 2" L1 L3 L4

echo "[$(date +%H:%M:%S)] ── ALL DONE ── (log: $LOG)"
