# PolicyEval

Unified policy rollout driver for the LIBERO-PerM benchmark. Runs any VLA
model that speaks the openpi websocket protocol in the P_Bench LIBERO env,
and writes SR + low-preference rewards + rollout HDF5s (for downstream VLM
compliance eval) into a uniform directory tree.

**Each run is driven by a runfile** (`/data3/zhixu/LIBERO-PeRM/VLA_infer/run_profile/ZeroShot/<name>.yaml`).
One runfile = one model × one task_suite. Multi-model sweeps = multiple
runfiles against multiple server ports.

> **Where things live:** the rollout *code* (`eval.py` etc.) stays here under
> P_Bench (env-side). The *launcher* (`run_client.sh`) and all *runfiles*
> live in `/data3/zhixu/LIBERO-PeRM/VLA_infer/` (run/inference-side). P_Bench/LIBERO
> owns data generation + LIBERO env; VLA_infer owns model serving + run
> orchestration; VLA repos themselves are cloned under `/data3/zhixu/LIBERO-PeRM/VLAs/`.

TL;DR entry point:
```
bash /data3/zhixu/LIBERO-PeRM/VLA_infer/run_client.sh run_profile/ZeroShot/smoke_pi05.yaml
```

## Layout

```
PolicyEval/
├── eval.py                  # the only entry point: runfile → sweep → rollouts
├── backends/                # inference backends (websocket-based)
├── low_pref_reward.py       # motion preference reward accumulator
└── hdf5_writer.py           # rollout HDF5 writer (P_Bench schema)
```

## Output tree

```
/data3/zhixu/LIBERO-PeRM/Policy_output/{exp_type}/{task_suite}/{experiment_name}/
                                 {model}/{condition}/{mission_id}/
├── HDF5/
│   ├── rollout_<ep>.hdf5
│   └── fail_case/rollout_<ep>.hdf5
├── Videos/
│   ├── Ep<ep>_success.mp4
│   └── fail_case/Ep<ep>_failure.mp4
├── init_states/
│   └── <bddl_base>_<0..14>.init
├── prompt.txt
└── rollout_summary.yaml    # sr, per-episode metadata, low-pref reward
```

`exp_type ∈ {0shot, lora_ft}`, `condition ∈ {base, pref}`.
`task_suite` resolves from `../configs/task_suites.yaml`.

## Running

### 1. Start the inference server (own conda env)

Pi0.5 (conda env `pi`):
```bash
cd /data3/zhixu/LIBERO-PeRM
conda activate pi
bash run_libero_server.sh
```

### 2. Run the client (conda env `libero_DataGen`)

```bash
cd /data3/zhixu/LIBERO-PeRM
bash run_client.sh run_profile/ZeroShot/smoke_pi05.yaml            # smoke
bash run_client.sh run_profile/ZeroShot/full_pi05.yaml             # full sweep
bash run_client.sh run_profile/ZeroShot/smoke_pi05.yaml --dry_run  # list cells
```

For an ad-hoc single mission, write a runfile with `missions: [1-1_004]` and
`num_trials: 1`. There is no separate single-mission CLI — every eval is a
sweep, even a sweep of size 1.

## Websocket contract (backend ⇔ driver)

Request (msgpack via `openpi_client.msgpack_numpy`):
```
observation/image        uint8 [H, W, 3]   agentview, 224x224
observation/wrist_image  uint8 [H, W, 3]   eye-in-hand, 224x224
observation/state        float32 [8]       eef_pos(3) + axisangle(3) + gripper(2)
prompt                   str               already condition-composed
```

Response:
```
actions  float32 [N, 7]    N >= replan_steps
```

Any backend conforming to this schema plugs in via `backends/`.

## Low-pref reward accumulation

Reads `low_prefs` from `mission_info.yaml`, instantiates matching
`BaseMotionPreference` objects via `Preference.pref_gen.create_motion_preference`,
and calls `pref.get_reward(env)` per step during `condition=pref` rollouts.
Summary (mean + satisfaction %) lands in `rollout_summary.yaml` per episode.
