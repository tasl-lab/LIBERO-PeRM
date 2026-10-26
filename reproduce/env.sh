#!/usr/bin/env bash
# Shared environment for the paper-reproduction scripts. Source, don't run.
# Every path below can be overridden from the caller's environment.

PERM_ROOT="${PERM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PERM_ROOT

export PERM_INFER="${PERM_ROOT}/VLA_infer"                 # server_*.sh, run_client.sh, run_profile/
export PERM_TRAIN="${PERM_ROOT}/VLA_train/openpi"           # runme_{gen_finetune_data,cal_stats,finetune}_<EXP>.sh
export PERM_PROFILE="${PERM_INFER}/run_profile"
export PERM_LOG_ROOT="${PERM_LOG_ROOT:-${PERM_ROOT}/reproduce/logs}"

# conda env used by the rollout client (LIBERO env + PolicyEval)
export CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-libero_DataGen}"

# how long to wait for a policy server to open its port (seconds)
export PERM_SERVER_TIMEOUT="${PERM_SERVER_TIMEOUT:-900}"

# DRY_RUN=1  → print every command instead of running it
# SKIP_SERVER=1 → assume the policy server is already up (you launched it yourself)
export DRY_RUN="${DRY_RUN:-0}"
export SKIP_SERVER="${SKIP_SERVER:-0}"
