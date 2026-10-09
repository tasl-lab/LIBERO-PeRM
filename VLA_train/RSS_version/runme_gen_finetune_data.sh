#export CUDA_VISIBLE_DEVICES=3,4
export CUDA_VISIBLE_DEVICES=6,7
python /data3/zhixu/P_Bench/VLA_train/post_training_script/convert_libero_data_to_lerobot.py \
  --output_dir /data4/zhixu/finetune_data \
  --repo_id perm/Mis_m1_to_m52 \

cd /data3/zhixu/P_Bench/VLAs
uv run openpi/scripts/compute_norm_stats.py --config-name pi05_libero_perm_m52



#  --repo_id perm/LE_0 \
#--repo_id perm/LE_1 \
#--repo_id perm/LE_2 \
#--repo_id perm/LE_3 \

#python post_training_script/convert_libero_data_to_lerobot.py \
#  --hdf5-path hdf5/put_white_mug_on_right_plate_with_handle.hdf5 \
#  --output-dir ./finetune_data/ \
#  --repo-id perm/min_veri \
#  --hdf5-path ./put_the_white_mug_on_right_plate.hdf5 \

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite