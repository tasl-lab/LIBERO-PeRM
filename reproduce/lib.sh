#!/usr/bin/env bash
# Helper functions for the reproduction scripts. Source after env.sh.
#
#   run_eval  <runfile.yaml>        launch the matching policy server, wait for its port,
#                                   run the rollout client (SR + SA → evalresult.xlsx), stop server
#   train_exp <EXP> [stages]        gen_finetune_data → cal_stats → finetune for VLA_train/openpi/runme_*_<EXP>.sh
#                                   stages: comma list from {data,stats,train}; default all that exist
#   log_section <text>

mkdir -p "${PERM_LOG_ROOT}"

log_section() { printf '\n[%s] ── %s ──\n' "$(date +%H:%M:%S)" "$*"; }

_run() {                      # run or echo a command depending on DRY_RUN
  if [[ "${DRY_RUN}" == "1" ]]; then echo "  (dry) $*"; else "$@"; fi
}

_yaml_server_field() {        # _yaml_server_field <runfile> <key>
  # shellcheck disable=SC1090
  source "${CONDA_SH}" && conda activate "${CONDA_ENV}" >/dev/null 2>&1
  python - "$1" "$2" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("server") or {}).get(sys.argv[2], ""))
PY
}

_server_script_for() {        # map server.vla → VLA_infer/server_*.sh
  case "$1" in
    pi05|pi0)     echo "${PERM_INFER}/server_pi05.sh" ;;
    openvla_oft)  echo "${PERM_INFER}/server_openvla_oft.sh" ;;
    groot)        echo "${PERM_INFER}/server_groot.sh" ;;
    smolvla)      echo "${PERM_INFER}/server_smolvla.sh" ;;
    univla)       echo "${PERM_INFER}/server_univla.sh" ;;
    *) echo "unknown server.vla='$1'" >&2; return 1 ;;
  esac
}

_wait_port() {                # _wait_port <port> <timeout_s>
  local port="$1" left="$2"
  while (( left > 0 )); do
    (echo > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null && return 0
    sleep 5; left=$(( left - 5 ))
  done
  return 1
}

run_eval() {
  local runfile="$1"
  [[ -f "${runfile}" ]] || runfile="${PERM_PROFILE}/${runfile}"
  [[ -f "${runfile}" ]] || { echo "runfile not found: $1" >&2; return 1; }
  local tag; tag="$(basename "${runfile}" .yaml)"
  local vla port srv
  vla="$(_yaml_server_field "${runfile}" vla)"
  port="$(_yaml_server_field "${runfile}" port)"
  srv="$(_server_script_for "${vla}")" || return 1
  log_section "eval ${tag}  (vla=${vla} port=${port})"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  (dry) bash ${srv} ${runfile}   # server, background"
    echo "  (dry) bash ${PERM_INFER}/run_client.sh ${runfile}"
    return 0
  fi

  local spid=""
  if [[ "${SKIP_SERVER}" != "1" ]]; then
    bash "${srv}" "${runfile}" > "${PERM_LOG_ROOT}/${tag}_server.log" 2>&1 &
    spid=$!
    if ! _wait_port "${port}" "${PERM_SERVER_TIMEOUT}"; then
      echo "  server on :${port} did not come up in ${PERM_SERVER_TIMEOUT}s — see ${PERM_LOG_ROOT}/${tag}_server.log" >&2
      kill "${spid}" 2>/dev/null || true
      return 1
    fi
  fi

  bash "${PERM_INFER}/run_client.sh" "${runfile}" 2>&1 | tee "${PERM_LOG_ROOT}/${tag}_client.log"
  local rc=${PIPESTATUS[0]}

  if [[ -n "${spid}" ]]; then
    kill "${spid}" 2>/dev/null || true
    pkill -f "serve.py.*$(basename "${runfile}")" 2>/dev/null || true
    sleep 5
  fi
  return "${rc}"
}

train_exp() {
  local exp="$1" stages="${2:-data,stats,train}"
  log_section "train ${exp}  (stages: ${stages})"
  local s script
  for s in ${stages//,/ }; do
    case "$s" in
      data)  script="${PERM_TRAIN}/runme_gen_finetune_data_${exp}.sh" ;;
      stats) script="${PERM_TRAIN}/runme_cal_stats_${exp}.sh" ;;
      train) script="${PERM_TRAIN}/runme_finetune_${exp}.sh" ;;
      *) echo "unknown stage '$s'" >&2; return 1 ;;
    esac
    if [[ ! -f "${script}" ]]; then
      echo "  [skip] ${s}: $(basename "${script}") not present (see header of runme_finetune_${exp}.sh for what it reuses)"
      continue
    fi
    _run bash "${script}" 2>&1 | tee -a "${PERM_LOG_ROOT}/train_${exp}.log"
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "  ${s} failed for ${exp}" >&2; return 1; }
  done
}
