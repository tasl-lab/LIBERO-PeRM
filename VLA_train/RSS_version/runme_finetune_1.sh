#export CUDA_VISIBLE_DEVICES=3,7
export CUDA_VISIBLE_DEVICES=4,5
#export CUDA_VISIBLE_DEVICES=5,6
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



#
#python post_training_script/convert_libero_data_to_lerobot.py \
#  --output_dir /data4/zhixu/finetune_data \
#  --repo_id perm/LE_1 \
#
#uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_LE1




export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
cd /data3/zhixu/P_Bench/VLAs
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_LE2 --exp-name=pi05_libero_perm_LE2 --overwrite

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
#uv run openpi/scripts/train.py pi05_libero_min_verify \
#  --exp-name=perm_min_veri \
#  --overwrite

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_lora --exp-name=pi05_lora_repo --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=pi0_lora_repo --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero --exp-name=pi05_repo --overwrite

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_min_veri_v1 --exp-name=pi05_libero_perm_min_veri_v2 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_v3 --exp-name=pi05_libero_perm_v3_t1 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_lts_v1 --exp-name=pi05_libero_perm_lts_v1 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_mis22 --exp-name=pi05_libero_perm_52_run02 --resume
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_LE0 --exp-name=pi05_libero_perm_LE0 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_LE1 --exp-name=pi05_libero_perm_LE1 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_LE2 --exp-name=pi05_libero_perm_LE2 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi/scripts/train.py pi05_libero_perm_LE3 --exp-name=pi05_libero_perm_LE3 --overwrite


