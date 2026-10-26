#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Series-7 ComprePref eval — S1 ONLY (S2 skipped 2026-05-06).
# Single-GPU per server, 2-track parallel.
#   Track A (GPU 4): standard → sequential
#   Track B (GPU 7): mix
# Each ckpt: server up → S1 client → server down → next.
# Logs: VLA_infer/logs/comp/<name>.log
# ────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/comp"
mkdir -p "${LOG_DIR}"
RUN_ROOT="${SCRIPT_DIR}/run_profile/LoRA/Pi05/Comp"

wait_port_open() {
  local port="$1" tries=180
  while (( tries-- > 0 )); do
    (echo > /dev/tcp/127.0.0.1/${port}) 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

run_ckpt() {
  local alias="$1" port="$2" s1="$3"
  local s1_rf="${RUN_ROOT}/${s1}"
  local srv_log="${LOG_DIR}/server_${alias}.log"
  local s1_log="${LOG_DIR}/${s1%.yaml}.log"

  echo "[${alias}] starting server :${port}  (log=${srv_log})"
  ( bash "${SCRIPT_DIR}/server_pi05.sh" "${s1_rf}" >"${srv_log}" 2>&1 ) &
  local srv_pid=$!
  echo "[${alias}] server pid=${srv_pid}"

  if ! wait_port_open "${port}"; then
    echo "❌ [${alias}] port ${port} not open in 900s — abort track" >&2
    kill -KILL "${srv_pid}" 2>/dev/null || true
    return 2
  fi
  sleep 30   # jit warm-up

  echo "[${alias}/S1] client → $(basename "${s1_rf}")"
  bash "${SCRIPT_DIR}/run_client.sh" "${s1_rf}" >"${s1_log}" 2>&1
  local rc=$?
  (( rc != 0 )) && echo "⚠ [${alias}/S1] rc=${rc}" || echo "✓ [${alias}/S1] done"

  echo "[${alias}] kill server pid=${srv_pid}"
  kill -KILL "${srv_pid}" 2>/dev/null || true
  pkill -KILL -P "${srv_pid}" 2>/dev/null || true
  sleep 5
}

# ── Track A: GPU 4 — standard → sequential (S1 only) ──
( run_ckpt standard   8021 Comp_S1_pi05_singpref_lora_step45k.yaml ; \
  run_ckpt sequential 8022 Comp_S1_pi05_libero_singpref_lora_step45k.yaml \
) > "${LOG_DIR}/track_A.log" 2>&1 &
PID_A=$!

# ── Track B: GPU 7 — mix (S1 only) ──
( run_ckpt mix 8023 Comp_S1_pi05_singpref_mix_lora_step45k.yaml \
) > "${LOG_DIR}/track_B.log" 2>&1 &
PID_B=$!

echo "════════════════════════════════════════════════════════════"
echo "Track A (GPU4, standard→sequential) pid=${PID_A}  log=${LOG_DIR}/track_A.log"
echo "Track B (GPU7, mix)                pid=${PID_B}  log=${LOG_DIR}/track_B.log"
echo "════════════════════════════════════════════════════════════"

wait "${PID_A}" "${PID_B}"
echo "All tracks finished."
