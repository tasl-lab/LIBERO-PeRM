# batch_run — unified mission batch runner

Three modes, each reads its own mission list + batch size from `batch_run.yaml`.

| Mode | mission list key | batch size key | default | 视频 | 用途 |
|------|------------------|----------------|---------|------|------|
| `dev`      | `dev_mission_ids`      | `dev_batch_size`      | 5  | 全存 | 开发期快速验证 layout / goal / task_plan |
| `test`     | `test_mission_ids`     | `test_batch_size`     | 5  | 全存 | push 前 smoke test 所有已就绪 mission |
| `data_gen` | `data_gen_mission_ids` | `data_gen_batch_size` | 50 | 前 3 个 | 正式出训练 HDF5 |

## 用法

```bash
cd LIBERO
python batch_run/batch_run.py                    # 读 yaml 里的 `mode`
python batch_run/batch_run.py --mode test        # CLI 覆盖
python batch_run/batch_run.py --mode data_gen
```

## batch_run.yaml 字段

```yaml
mode: dev                      # dev | test | data_gen （CLI --mode 可覆盖）
dev_batch_size: 5              # 每个 mission rollout 数（dev 模式）
test_batch_size: 5             # test 模式
data_gen_batch_size: 50        # data_gen 模式
overwrite_old: 1               # 1=覆盖已有 HDF5, 0=跳过已存在
use_forcibly: 1                # 1=强制生成最终合集 HDF5, 0=低 SR 时跳过
parallel_execution: True       # True=ThreadPool 并行跑；False=生成 .sh 脚本
workers: 16                    # 最大并发 mission 数（单 GPU OOM 风险：>8 需留意）
tasksuite: SinglePref          # 结果进 VLA_data/<tasksuite>/Mission_*.hdf5
dev_mission_ids:
  - 2-11_116
  - 2-12_009
data_gen_mission_ids:
  - 1-1_004
test_mission_ids:
  - 1-1_004
  - 1-1_006
```

## 行为

- **parallel_execution=True**：ThreadPoolExecutor 直接并行跑，LiveDashboard 显示每个 mission 的 SR。日志落 `parallel_logs/<mission_id>.log`，SR 汇总落 `batch_run/sr_summary.txt`。
- **parallel_execution=False**：
  - `dev` / `data_gen` → 生成 `batch_run.sh`
  - `test` → 生成 `test_before_push.sh`（每个 mission 独立 pass/fail，最终汇总 exit code）

## data_gen 额外行为

- `--n_save_video 3` 自动加上：每个 mission 只存前 3 个 rollout 的 MP4（50 rollout × 5 MB 太占盘）
- rollout 生成完会调 `concatenate_hdf5_demos` 合并到 `VLA_data/<tasksuite>/Mission_<id>.hdf5`
- 合并后源 `rollout_*.hdf5`（含 `fail_case/`）被删掉

## 关联文件

- `main.py` — 单 mission 入口，被本脚本以 subprocess 调
- `configs/task_suites.yaml` — 训练 suite 的 source of truth（加新 mission 要同步进来）
- `scripts/cleanup_policy_hdf5.py` — 清 `Policy_output` 残留 rollout HDF5
