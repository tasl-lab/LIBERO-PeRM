#!/usr/bin/env bash
# Chain B: L1+L3 → L4 on GPU 0 + 1
set -uo pipefail
INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer
PROFILE=$INFER/run_profile/LoRA/Pi05
LOG=${LOG_DIR:-/tmp/pscale_pref_ood_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG"
echo "[B] Logs → $LOG"

run_iter() {
    local C1=$1 C2=${2:-} P=$3
    echo "[B $(date +%H:%M:%S)]   ${C1}${C2:+ + $C2} × $P"
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

for P in speed left red multi; do run_iter L1 L3 "$P"; done
for P in speed left red multi; do run_iter L4 ""  "$P"; done

echo "[B $(date +%H:%M:%S)] DONE"
