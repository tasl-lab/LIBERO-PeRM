#!/usr/bin/env bash
# Monitor perm RLDS tfds build progress.
# Usage: bash check_progress.sh

LOG=/tmp/claude-1008/-data3-zhixu-P-Bench/9b4cbebd-698c-44c3-8e9b-33b3c3f32b4e/tasks/b450grqt8.output
OUT_DIR=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm_rlds_smoke/perm/1.0.0

echo "=== process ==="
pgrep -af "tfds build perm" | head

echo ""
echo "=== log tail (last 25 lines) ==="
[ -f "$LOG" ] && tail -25 "$LOG" || echo "log not found: $LOG"

echo ""
echo "=== output dir ==="
if [ -d "$OUT_DIR" ]; then
  du -sh "$OUT_DIR" 2>/dev/null
  ls -lh "$OUT_DIR" | head -20
  echo ""
  N_SHARDS=$(ls "$OUT_DIR"/perm-train.tfrecord-* 2>/dev/null | wc -l)
  echo "tfrecord shards: $N_SHARDS"
else
  echo "(no output dir yet)"
fi

echo ""
echo "=== incomplete dir ==="
INC=$(find "$(dirname "$OUT_DIR")"/.. -maxdepth 4 -type d -name "*.incomplete*" 2>/dev/null | head -3)
[ -n "$INC" ] && echo "$INC" || echo "(none)"
