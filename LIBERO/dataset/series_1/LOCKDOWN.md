# Motion-Preference Lockdown（暂定版）

人工过审后冻结的 metric / layout / demo 版本。**改动这里列的任何一项都要重跑标定 + 重渲对比视频 + 重新过审。**

审核方式：`/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/` 下的 pref-vs-no-pref
对比视频（上排 pref、下排 naive、右侧逐步 satisfaction 曲线 + metric 公式说明）。
过审的 mission 视频移回各自 family 目录。

## 状态

| family | preference | 状态 |
|---|---|---|
| 1-1 speed_up | Speed up | **已锁**（2026-09-01 过审） |
| 1-2 slow_down | Slow down | **已锁**（2026-09-01 过审） |
| 1-4 avoid | Keep away from X | **已锁**（2026-09-01 过审） |
| 1-6 straight | Move straight | **已锁**（2026-09-01 过审） |
| 1-7 move_higher | Move higher | **已锁**（2026-09-01 过审） |
| 1-8 rotation | Rotate when moving | **已锁**（2026-09-01 过审） |
| 1-5 close_to_ground | Close to ground | **已锁**（2026-09-01 过审） |
| 1-3 acc | Avoid sudden acceleration | **已锁**（2026-09-01 过审） |

八家全部过审。

## 已锁家族的 metric 定义

**1-1 / 1-2 speed（per-subtask 计时）**
- 分段：`TaskPlanCompliance` 两阶段实时判定。pick = 夹爪接触抓住 + 抬起；
  put = 物体到目标附近 + 松爪；BDDL 总成功兜底收口。
- `R_i` = 第 i 个 subtask 的步数；`P_i` = pref demo 在该段的中位步数。
- `score_i = clip(P_i / R_i, 0, 1)`（speed_up）/ `clip(R_i / P_i, 0, 1)`（slow_down），
  方向由 `mean(subtask_pref_anchor) <= mean(subtask_naive_anchor)` 自动推断。
- `satisfaction = mean_i(score_i) x 100`。曲线：每完成一个 subtask 出一个点、点间直线连接。
- 注意：这个口径下 naive 不会到 0（只慢 ~1.4x → 72 分），图里的 30 gate 对 speed 家族不适用。

**1-4 avoid（carry 段最大水平距离）**
- `raw = MAX over carry segment of min(d(EE, target), d(carried obj, target))`，
  d 只算水平（xy），carry 段 = [抓起, 最终下放开始]。
- `satisfaction = clip((raw - N) / (P - N), 0, 1) x 100`。

**1-6 straight（carry 段路径冗余）**
- `raw = path_length / chord - 1`（0 = 完全直线），取 carry 段。
- 同上的线性双锚映射。

**1-7 move_higher（横移段最高点）**
- `raw = MAX EE z over sustained lateral-move steps - robot base z`。
- 同上的线性双锚映射。gen 侧 `height_offset = +0.10`，move_to 的 z cap 对这一家
  额外放宽 `+height_offset`，免得把 pref 信号压没。

**1-3 acc（单步加速度的 P95）**
- `a_t = |p(t+1) − 2p(t) + p(t−1)|`，p = EE 世界坐标，单位 m/step²（不换算成秒，
  控制频率固定，换算只是乘常数）。
- `raw = P95(a_t)`，t 取 `[开局, 首次 success)`。整段 episode 一个数。
- **P95 而不是 MAX**：MAX 被抓取 / 松爪那一下的接触尖峰占住，pref 和 naive 一样大，
  分不开（107 d=2.0、109 d=1.1）；P95 落在第 7-8 大的那个值上，量的是真实移动的
  抖动（107 d=10.7、109 d=9.4）。
- **样本门槛 20**：不足 20 个 a_t 不出分。P95 在样本 <20 时就等于"几个数里的最大"，
  会让 running 曲线开头 100→0 乱跳。终点数值不受影响（真实 episode 有 130+ 个样本）。
- **曲线怎么读**：视频里画的是同一公式在 `[0, t]` 上重算，不是"此刻多抖"。
  百分位在样本增长时会把早期尖刺挤出 95 分位，所以曲线会往上漂、naive 那条呈锯齿
  （出一次尖刺掉一次，然后被稀释爬回去）。**只有终点是 metric。**

**1-5 close_to_ground（横移段最高点，跟 1-7 同一个 raw、方向相反）**
- `raw = MAX EE z over sustained lateral-move steps - robot base z`，跟 1-7 完全同一个量。
- 方向靠双锚的 `sign(P - N)` 自动编码：这家 P < N（越低越好），1-7 是 P > N。公式不用改。
- gen 侧 `height_offset`：070 = **−0.12**，114 = **−0.18**。−0.22 试过，两个 mission
  双双 0/40（撞桌面），这是当前的地板。

**1-8 rotation（横移中的绝对转角）**
- `raw = SUM of EE 转角 over 有横移的 step`（rad）。横移判定 = 单步水平位移 > 1mm。
- 同上的线性双锚映射。
- **不是**旧的比值 `on_lat / total`：统一 layout 后 naive 也几乎只在横移时转，
  份额贴着 pref（061 pref 0.775 vs naive 0.743，cohen_d 1.67 且 raw 范围重叠）。
  绝对量才是 preference 的字面意思，061 d=35.9 / 114 d=98.9，两边都无重叠。

## 冻结的锚值

| mission | metric | P (pref) | N (naive) | demo 数 |
|---|---|---|---|---|
| 1-1_004 | speed_range | 232.5 | 304.5 | pref 10 / naive 10 |
| 1-1_006 | speed_range | 266.5 | 370.5 | pref 10 / naive 10 |
| 1-2_065 | speed_range | 305.5 | 160.5 | pref 100 / naive 10 |
| 1-2_113 | speed_range | 414.0 | 209.0 | pref 100 / naive 10 |
| 1-4_034 | avoid_to_object | 0.306884 | 0.186322 | pref 10 / naive 10 |
| 1-4_053 | avoid_to_object | 0.441327 | 0.253846 | pref 10 / naive 10 |
| 1-6_071 | straight_path | 0.023027 | 0.337935 | pref 100 / naive 10 |
| 1-6_105 | straight_path | 0.043001 | 0.383314 | pref 100 / naive 10 |
| 1-7_015 | height_offset | 0.422879 | 0.313712 | pref 10 / naive 10 |
| 1-7_104 | height_offset | 0.328841 | 0.221220 | pref 100 / naive 10 |
| 1-5_070 | height_offset | 0.171962 | 0.271424 | pref 10 / naive 10 |
| 1-5_114 | height_offset | 0.190481 | 0.373142 | pref 10 / naive 10 |
| 1-8_061 | rotation | 0.724473 | 0.150798 | pref 10 / naive 10 |
| 1-8_114 | rotation | 1.531738 | 0.406565 | pref 100 / naive 10 |
| 1-3_107 | acc_range | 0.002465 | 0.004246 | pref 100 / naive 10 |
| 1-3_109 | acc_range | 0.001911 | 0.003753 | pref 10 / naive 10 |

speed 家族的 per-subtask 锚（步数）：

| mission | subtask_pref_anchor | subtask_naive_anchor |
|---|---|---|
| 1-1_004 | [41.0, 72.5, 42.5, 73.5] | [44.5, 95.5, 75.0, 86.0] |
| 1-1_006 | [42.0, 92.5, 61.5, 70.0] | [67.5, 131.5, 73.5, 95.0] |
| 1-2_065 | [98.0, 205.0] | [54.0, 106.0] |
| 1-2_113 | [150.0, 264.0] | [76.0, 131.0] |

## 冻结的 gen 侧常数

| 常数 | 值 | 作用 |
|---|---|---|
| `EE_Z_CAP_REL_M` | 0.30 | 桌面场景 gripper z 上限（相对支撑面） |
| `EE_Z_CAP_REL_FLOOR_M` | 0.45 | floor 场景同上 |
| 端点豁免 | 本段指令终点 −0.05 | cap 不压任务本身需要的高度 |
| `DEFAULT_MOVE_TO_HEIGHT` | 0.20 | move_to 巡航垫高 |
| cabinet_top bump | 0.12 | 去顶层抽屉的额外垫高（原 0.20） |
| avoid 绕行 clearance | `max(0.3, threshold)` | 1-4_034 threshold 0.45 / 1-4_053 0.45 |
| `PERM_INIT_RANGE_SCALE` | 1.0（关） | init 随机化范围倍数，本次不启用 |
| `PERM_INIT_YAW_JITTER` | 0.0（关） | init yaw 抖动，本次不启用 |

支撑面 = 开局所有 free-joint 物体 body z 的 **median**（median 而非 min，避免被掉到桌外的
物体拉穿——1-1_006 的 milk 初始 z=−0.03 曾把整个场景误判成 floor）。

## 冻结的 layout 改动

- **1-4_034**：wine_bottle init region 挪到 `(-0.105 -0.025 -0.095 -0.015)`（抽屉南侧，
  贴 naive 路径中段）。规则：障碍放 naive 路径中段旁、离碗轨迹 ≥ 接触线+2.5cm、
  离一切 fixture 占位（含打开的抽屉）≥ 瓶半径+2cm、放路径**西**侧。
- **1-4_053**：layout 不动，只把 avoid threshold 从 0.15 提到 0.45。
- **1-8_061 / 1-8_114**：naive 不再用各自的 `no_pref_layout/` bddl，改成跟 pref 同一份
  （`--use_pref_layout`），再用 `--pair_init_from` 把 pref 每条 demo 的 `states[0]` 注入
  naive 对应那条。核实：两边物体开局位置差 0.00mm、朝向一致、资产名集合相同。
  注入必须走 wrapper 的 `env.set_init_state()`（裸 `sim.set_state_from_flattened` 会被
  wrapper 的 observable 缓存弹回，实测残留 16mm 误差）。
  这两个是全 16 个 mission 里唯二带过双 layout 的，现在双 layout 作废。

## demo 数据位置

- pref：`/data3/zhixu/PeRM_data/demos/SinglePref/Mission_<id>.hdf5`
  （repo 侧 `VLA_data/SinglePref/` 是软链）
- naive：`/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref_naive/Mission_<id>.hdf5`（实体文件）
- 生成侧失败的 rollout 进 fail_case、不进合并文件，所以存下来的每条都是成功 demo。
