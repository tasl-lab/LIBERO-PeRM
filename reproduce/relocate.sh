#!/usr/bin/env bash
# The runfiles / training scripts carry absolute paths from the machine the paper was run on
# (/data3/zhixu/LIBERO-PeRM). Run this once after cloning elsewhere to rewrite them to $PERM_ROOT.
#   bash reproduce/relocate.sh            # rewrite in place
#   bash reproduce/relocate.sh --check    # only list files that still carry the old prefix
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${HERE}/env.sh"
OLD="/data3/zhixu/LIBERO-PeRM"
[[ "${PERM_ROOT}" == "${OLD}" ]] && { echo "PERM_ROOT already is ${OLD}; nothing to do"; exit 0; }
mapfile -t FILES < <(grep -rl --include='*.yaml' --include='*.sh' --include='*.py' "${OLD}" \
  "${PERM_ROOT}/VLA_infer" "${PERM_ROOT}/VLA_train/openpi" "${PERM_ROOT}/VLA_train/post_training_script" "${HERE}/configs")
if [[ "${1:-}" == "--check" ]]; then printf '%s\n' "${FILES[@]}"; exit 0; fi
for f in "${FILES[@]}"; do sed -i "s|${OLD}|${PERM_ROOT}|g" "$f"; done
echo "rewrote ${#FILES[@]} files: ${OLD} -> ${PERM_ROOT}"
