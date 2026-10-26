# run_profile/ — 实验 runfile 仓库

每个 runfile 描述一次可重复的 client 运行：跑哪些 mission、打哪个 server、期望对方是什么模型。
按实验阶段分目录：

```
run_profile/
├── ZeroShot/       # 0-shot 评估（pi0 / pi0.5 / openvla-oft / ...）
└── LoRA/           # LoRA finetune 后的评估（TODO）
```

## 运行方式

```bash
# 1. 先起 server（在另一个 terminal / 另一个 conda env）
cd /data3/zhixu/LIBERO-PeRM
bash run_libero_server.sh             # pi0.5 on :8000（默认）

# 2. 再跑 client（libero_DataGen env）
cd /data3/zhixu/LIBERO-PeRM
bash run_client.sh run_profile/ZeroShot/smoke_pi05.yaml
```

## Runfile schema（ZeroShot 示例）

```yaml
experiment_name: pi05_singlepref_smoke     # 必填；作为输出路径中的一段
exp_type: 0shot                             # 0shot | lora_ft
task_suite: single-pref                     # 来自 configs/task_suites.yaml

missions:                                   # 可选；不给 / 为空 → 自动取 task_suite 全量
  - 1-1_004
  - 2-1_019

conditions: [base, pref]
num_trials: 2
replan_steps: 5
max_steps: null
seed: 7

server:
  name: pi05                                # 输出路径里的 {model} 段
  host: 0.0.0.0
  port: 8000
  expected_model: pi05                      # 不匹配 → 启动时立刻 abort
  expected_checkpoint: null                 # 可选：checkpoint_dir 的子串匹配
```

### 字段说明

| 字段 | 作用 |
|---|---|
| `experiment_name` | 写进输出路径 `rollouts/{exp_type}/{task_suite}/{experiment_name}/{model}/{condition}/{mission}/` |
| `missions` | 显式子集；省略则从 `configs/task_suites.yaml` 的 `task_suite` 全取 |
| `server.expected_model` | 连上 server 后拿 metadata 做子串匹配，不过不匹配就 `ServerMismatchError` 退出，不写任何文件 |
| `server.expected_checkpoint` | 可选的 checkpoint_dir 子串校验，用来锁定具体 ckpt（例如 `"LE0/10000"`） |

## 多 server / 多模型

**一个 runfile = 一个 model × 一个 task_suite。**
并发跑两个模型 = 起两个 server（不同端口）+ 两个 runfile + 两个 `run_client.sh`。
好处：
- 每个 run 独立 log，独立可 rerun
- server 端可以在不同 GPU / 不同 conda env（openpi 的 `pi` vs openvla-oft 的 `openvla-oft`）
- 不会因为某个 server 挂了影响其他的

## Server metadata 约定

为了让 `expected_model` 校验有根据，每个 server 必须在启动时在 metadata 里至少塞以下键之一：

| 键 | 例子 |
|---|---|
| `config_name` | `"pi05_libero_pbench_LE0"` |
| `model` | `"openvla_oft"` |
| `checkpoint_dir` | `"/data3/zhixu/LIBERO-PeRM/checkpoints/pi05_libero_pbench_LE0/.../10000/"` |

openpi 的 `scripts/serve_policy.py` 已经打过补丁会自动塞 `config_name` + `checkpoint_dir`。
新增 backend 时（openvla_oft / groot / ...）记得在 `serve.py` 里把 metadata 填好。
