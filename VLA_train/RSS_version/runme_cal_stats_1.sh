#export CUDA_VISIBLE_DEVICES=3,7
export CUDA_VISIBLE_DEVICES=3,4
#export CUDA_VISIBLE_DEVICES=5,6
#export HF_LEROBOT_HOME=/data3/zhixu/P_Bench/finetune_data/ #already set in the bashrc

#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_min_veri_v1
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_v3
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_lora_v1
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_lts_v1
cd /data3/zhixu/P_Bench/VLAs
uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE0
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE1
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE2
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE3
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE4
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05
#uv run openpi/scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune