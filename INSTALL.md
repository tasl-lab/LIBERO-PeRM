# LIBERO-PeRM Installation Guide

## Prerequisites

- Ubuntu 22.04 (tested)
- NVIDIA GPU with driver installed (tested: RTX 6000 Ada)
- Conda (Miniconda or Anaconda)
- Git

## 1. Clone Repository

```bash
git clone https://github.com/tasl-lab/LIBERO-PeRM.git
cd LIBERO-PeRM
```

## 2. Create Conda Environment

```bash
conda create -n libero python=3.10 -y
conda activate libero
```

## 3. Install Dependencies

### Core packages

```bash
pip install numpy==1.22.4
pip install scipy==1.13.1
pip install h5py==3.14.0
pip install opencv-python==4.6.0.66
pip install pyyaml==6.0.3
pip install matplotlib
pip install gitpython
```

### MuJoCo

```bash
pip install mujoco==3.3.1
```

### Robosuite (robot simulation framework)

```bash
pip install robosuite==1.4.0
```

### OMPL (motion planning, optional - needed for drawer/side-pick skills)

```bash
pip install ompl==1.7.0
```

### LIBERO (task framework)

```bash
cd LIBERO
pip install -e .
cd ..
```

### PyTorch (optional - not required for data generation, may fail on some systems)

```bash
pip install torch==2.10.0
```

> Note: `import torch` may fail due to missing `libcusparseLt.so.0`. This does not affect the data generation pipeline.

### Teleoperation (optional - for keyboard control on local machine)

```bash
pip install pynput
```

## 4. GPU Rendering Setup

LIBERO-PeRM uses EGL for GPU-accelerated offscreen rendering. Verify EGL works:

```bash
cd LIBERO
conda run -n libero python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'
from libero.libero.envs import OffScreenRenderEnv
env = OffScreenRenderEnv(
    bddl_file_name='libero/libero/bddl_files/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl',
    camera_heights=256, camera_widths=256
)
env.reset()
print('EGL rendering OK')
env.close()
"
```

If you have multiple GPUs, set `EGL_DEVICE_ID` to select which one:

```bash
EGL_DEVICE_ID=0  # GPU 0
EGL_DEVICE_ID=1  # GPU 1
```

## 5. Verify Installation

### Run a single mission (headless, generates video + HDF5)

```bash
cd LIBERO
conda run -n libero python main.py \
  --mission_id "2-12_085" \
  --naive_llm_plan 0 --run_layout_gen 0 --high_pref_gpt_plan 0 \
  --low_pref_gpt_plan 0 --generate_evaluation_data 0 \
  --enable_collision_check 1 --batch_size 1 --batch_start 0 \
  --manually_set_offset 1 --check_init_layout 0 \
  --gen_in_batch 1 --overwrite_old 1 --use_forcibly 1 --verbose
```

Success output: `✅ Video saved: dataset/series_2/perm_2_12/libero_085/Videos/rollout_0.mp4`

### Run keyboard teleoperation (requires display)

```bash
cd LIBERO
python teleop_collector.py --mission_id 3-4_058_1
```

### Run skill parameter tuner

```bash
cd LIBERO
python skill_tuner.py --mission_id 2-4_112 --skill pick_top_angle --n_rollouts 1 --n_workers 1
```

## 6. Project Structure

```
LIBERO-PeRM/
├── LIBERO/                          # Main working directory
│   ├── main.py                      # Entry point for data generation
│   ├── step1_data_gen.py            # Data generation pipeline
│   ├── skill_tuner.py               # Parallel parameter search tool
│   ├── teleop_collector.py          # Keyboard teleoperation + grasp recording
│   ├── MotionGen/                   # Motion generation
│   │   ├── heuristic_skill/
│   │   │   └── heuristic.py         # All skill functions (pick, put, etc.)
│   │   ├── motion_planner/
│   │   │   └── ompl_planner.py      # OMPL motion planner
│   │   ├── motion_gen.py            # Skill dispatcher + pipeline
│   │   └── task_priority_osc.py     # Experimental OSC controller patch
│   ├── dataset/                     # Mission configs + generated data
│   │   └── series_<S>/perm_X_Y/libero_ZZZ/  # Per-mission directory
│   │       ├── mission_info.yaml    # Task config
│   │       ├── *.bddl              # Scene definition
│   │       ├── Videos/             # Generated videos (gitignored)
│   │       └── HDF5/              # Generated trajectories (gitignored)
│   ├── grasp_params/               # Recorded grasp poses (from teleop)
│   ├── libero/libero/              # LIBERO framework
│   ├── Preference/                 # Preference system
│   ├── TaskGen/                    # Task generation
│   └── Utility/                   # Shared helpers
├── VLA_data/                       # Demo HDF5 per suite + suite registries
├── VLA_train/                      # Fine-tuning scripts (openpi)
├── VLA_infer/                      # Policy servers, rollout client, run profiles
└── reproduce/                      # One script per table / figure of the paper
```

## 7. Common Issues

### EGL rendering fails

```
OpenGL.raw.EGL._errors.EGLError
```

Fix: Ensure NVIDIA driver is installed and EGL libraries are available:

```bash
sudo apt install libegl1-mesa-dev
```

### `free(): invalid pointer` on exit

This is a cleanup error in robosuite/MuJoCo EGL context. It does not affect data generation. Safe to ignore.

### Video files corrupted / can't open

Caused by `--overwrite_old 1` when previous video writer hasn't released. Wait for the previous run to fully complete before re-running.

### `ImportError: libcusparseLt.so.0`

PyTorch installation issue. Not needed for data generation. Ignore unless you need torch for evaluation.
