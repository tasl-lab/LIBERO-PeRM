#!/usr/bin/env bash
# Apply the LIBERO-PeRM openpi overlay onto a fresh openpi checkout.
#
# The overlay holds every local customization the eval/serve/train pipeline
# needs, snapshotted from the working openpi tree on 2026-07-25:
#   serve.py                                          (custom uniform server, untracked upstream)
#   src/openpi/training/config.py                     (all pi05_* custom TrainConfigs, +970 lines)
#   src/openpi/policies/policy_config.py              (small fix)
#   src/openpi/training/checkpoints.py                (small fix)
#   packages/openpi-client/.../websocket_client_policy.py  (keepalive fix)
#   scripts/serve_policy.py, scripts/compute_norm_stats.py, examples/libero/main.py
#   uv.lock                                           (pinned deps)
#   assets/                                           (norm_stats for all custom configs, 260K)
#
# Usage:
#   bash apply_overlay.sh [/path/to/openpi]     # default: ../../VLAs/openpi
#
# ⚠️  BEFORE re-downloading/deleting an openpi folder, move out or keep:
#     checkpoints/   (~498G trained ckpts — NOT in this overlay, irreplaceable)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$(readlink -f "$HERE/../../VLAs/openpi")}"

[[ -d "$TARGET/src/openpi" ]] || { echo "❌ $TARGET does not look like an openpi checkout (no src/openpi)"; exit 1; }

cp -rv "$HERE/overlay/." "$TARGET/" | sed "s|$HERE/overlay/|  + |"

echo
echo "✅ overlay applied to $TARGET"
echo "   next: cd $TARGET && GIT_LFS_SKIP_SMUDGE=1 uv sync   (then serve via VLA_infer/server_pi05.sh)"
