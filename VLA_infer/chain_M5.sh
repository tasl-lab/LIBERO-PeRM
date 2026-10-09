#!/usr/bin/env bash
# M5 chain: in-domain → layout-OOD → 4× pref-OOD on GPU 5 / port 8045.
set -uo pipefail
INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer
PROFILE=$INFER/run_profile/LoRA/Pi05
LOG=${LOG_DIR:-/tmp/pscale_M5_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG"
echo "[M5] Logs → $LOG"

run_yaml() {
    local TAG=$1 YAML=$2
    echo "[M5 $(date +%H:%M:%S)]   ── $TAG ──"
    bash $INFER/server_pi05.sh $YAML > "$LOG/${TAG}_server.log" 2>&1 &
    SP=$!
    sleep 90
    bash $INFER/run_client.sh $YAML > "$LOG/${TAG}_client.log" 2>&1
    kill $SP 2>/dev/null || true
    pkill -f "serve.py.*$(basename $YAML)" 2>/dev/null || true
    sleep 6
}

run_yaml indomain     $PROFILE/H_M5_4trial.yaml
run_yaml layout_ood   $PROFILE/H_M5_layout_ood.yaml
for P in speed left red multi; do
    run_yaml pref_ood_${P}  $PROFILE/H_M5_pref_ood_${P}.yaml
done
echo "[M5 $(date +%H:%M:%S)] DONE"
