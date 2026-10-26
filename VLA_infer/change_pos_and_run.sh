#!/bin/bash

num_rounds=15

for ((i=1; i<=num_rounds; i++))
do
  echo "🔁 Round $i: running random_change_pos.sh"
  bash random_change_pos.sh

  echo "▶️ Round $i: running run_libero_client.sh"
  bash run_libero_client.sh

  echo "✅ Done one round"
  echo
done
