#export CUDA_VISIBLE_DEVICES=3
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /data3/zhixu/P_Bench/VLAs/openpi
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH
source examples/libero/.venv/bin/activate
python examples/libero/main.py --args.use-init 0


#/data3/zhixu/P_Bench/VLAs/openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.bddl