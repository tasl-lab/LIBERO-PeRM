#!/usr/bin/env bash
# PrefScale pref-OOD eval — 28 (cell, pref) cells. Each starts its own server.
# Pairs run on GPU 4 + GPU 7 simultaneously per pref iteration.
set -uo pipefail

INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer
PROFILE=$INFER/run_profile/LoRA/Pi05
TS=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/pscale_pref_ood_${TS}
mkdir -p "$LOG"
echo "Logs → $LOG"

run_iter() {
    local C1=$1 C2=$2 P=$3
    echo "[$(date +%H:%M:%S)]   ── ${C1}${C2:+ + $C2} × pref=$P ──"

    bash $INFER/server_pi05.sh $PROFILE/H_${C1}_pref_ood_${P}.yaml > "$LOG/${C1}_${P}_server.log" 2>&1 &
    SP1=$!
    if [[ -n "$C2" ]]; then
        bash $INFER/server_pi05.sh $PROFILE/H_${C2}_pref_ood_${P}.yaml > "$LOG/${C2}_${P}_server.log" 2>&1 &
        SP2=$!
    fi
    sleep 90

    bash $INFER/run_client.sh $PROFILE/H_${C1}_pref_ood_${P}.yaml > "$LOG/${C1}_${P}_client.log" 2>&1 &
    CP1=$!
    if [[ -n "$C2" ]]; then
        bash $INFER/run_client.sh $PROFILE/H_${C2}_pref_ood_${P}.yaml > "$LOG/${C2}_${P}_client.log" 2>&1 &
        CP2=$!
        wait $CP1 $CP2
    else
        wait $CP1
    fi

    kill $SP1 ${SP2:-} 2>/dev/null || true
    pkill -f "serve.py.*H_${C1}_pref_ood_${P}" 2>/dev/null || true
    [[ -n "$C2" ]] && pkill -f "serve.py.*H_${C2}_pref_ood_${P}" 2>/dev/null || true
    sleep 6
}

run_pair() {
    local C1=$1 C2=${2:-}
    echo "[$(date +%H:%M:%S)] ── Pair: ${C1}${C2:+ + $C2} ──"
    for P in speed left red multi; do
        run_iter "$C1" "$C2" "$P"
    done
}

run_pair M1 M2
run_pair M3 M4
run_pair L1 L3
run_pair L4

echo "[$(date +%H:%M:%S)] ── ALL DONE ── (log: $LOG)"
