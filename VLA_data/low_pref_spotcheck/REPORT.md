# T2 Low spot-check：每类 low-level pref 各 1 条 naive + 1 条新 pref demo（2026-07-01）

生成：series-1 全 10 mission，naive 9 条新生成（1-2_113 复用 6-28 smoke 的）+ pref 10 条新生成。
SR 全 1.00，两批各 ~40s 跑完（GPU 6, MUJOCO_GL=egl）。
- naive: `VLA_data/SinglePref_naive/`（每 mission 1 demo）
- 新 pref: `VLA_data/SinglePrefSpot/`（临时 spot 目录，不碰 /data4 与 VLA_data/SinglePref）
- 视频: `VLA_data/low_pref_spotcheck/videos/`（20 条 mp4，agentview）
- 数表: `discrim_freshpref.csv`（新 pref vs naive, n=1）、`discrim_data4.csv`（/data4 老 GT vs naive, n=5）

## 每类判定

| 类型 | mission | naive raw | 新 pref raw | /data4 GT raw | 判定 |
|---|---|---|---|---|---|
| speed_range 慢 (0.5) | 1-2_113 | 211 步 | 416 | 414.4 | ✅ 分离大、方向对 |
| speed_range 快 (2.0) | 1-1_004 / 006 | 339 / 389 | 279 / 336 | 275.8 / 334.6 | ✅ 分离清晰、方向对 |
| height_offset 高 (+0.1) | 1-7_104 | 0.565 | 0.881 | 0.881 | ✅ 分离好 |
| height_offset 低 (-0.15) | 1-5_114 | 0.632 | 0.749 | 0.779 | ✅ 中等分离 |
| straight_path (0.05) | 1-6_105 | 0.237 | 0.266 | 0.267 | ⚠️ 方向对但 gap 小（~12%），pref 自身 raw 仅 0.27 |
| rotation (0.1) | 1-8_114 | 0.143 | 0.013 | 0.013 | ⚠️ metric 语义与 mission 相反（见下），但分离明显 |
| max_acc (0.03) | 1-3_107 / 109 | 0.052 / 0.022 | 与 naive **bitwise 相同** | ≈naive (0.049/0.026) | ❌ gen no-op，双锚 degenerate |
| avoid_object (0.15m) | 1-4_034 | 1.000 | 1.000 | 1.000 | ❌ metric raw 饱和，双锚 degenerate |

新 pref ≈ /data4 老 GT（所有类型 raw 几乎一致）→ **LIBERO-plus overlay 合并后 pref gen 路径没坏**。

## 3 个问题类型的机制

1. **max_acc = gen no-op**：pref 与 naive 的 ee 轨迹逐位相同（135/149 步完全一致）——gen 侧从未真正
   施加加速度限制（老 /data4 demo 同样如此）。raw 0.05 > max_acc 0.03，连 pref demo 自己都不满足偏好
   （sat 仅 ~5%）。修法 = gen 里实现真·加速度 clamp，或此类型放弃标定。
2. **avoid_object metric 饱和**：`AvoidToObjectPreference` per-step raw = `dist>threshold ? 1 : dist/threshold`，
   threshold 映射正确（0.15），但该 mission 里连 naive 直线轨迹都全程离 wine bottle >0.15m → 两边恒 1.0。
   pref demo 确实绕路了（263 vs 166 步，视频可见），metric 却看不见。修法 = raw 换连续量
   （如 trajectory min-distance 或 mean dist），或换个 naive 会路过瓶子的 layout。
3. **rotation metric 语义反**：mission = "Rotate when moving"（要转），`RotationPreference` 却 reward
   "descent 段 NO rotation"（角度越小分越高），yaml 的 `angle_per_step=0.1` 被重解释成 max-angle 阈值。
   pref demo 全程转 wrist → raw 0.013 低分；naive 不转 → 0.143。双锚公式 sign 自适应所以"能用"
   （标定后仍可判别），但它测的是"像不像 pref demo 的 descent 旋转量"，不是"移动时是否在转"。

## sat 列说明

当前 sat 用的还是 mission_info 里的旧 legacy 单锚（尚未 `calibrate --apply` 双锚），所以
speed_range 出现 naive sat=100 的失真——这正是双锚要修的问题，apply 后 pref≈100 / naive≈0。

## 建议下一步

- speed_range / height_offset / straight_path / rotation（4 类 8 mission）：可直接全量
  naive gen（每 mission ~10 条）→ `calibrate_low_pref.py --apply` → 区分度表（cohen_d 有 n 才可算）。
- max_acc：先修 gen（真的限制加速度）再标定。
- avoid_object：先改 metric raw（连续距离量）再标定。
- straight_path：可标定，但列入"demo-gen 优化候选"（pref demo 直线度本身不高）。

## 追记（2026-07-01 同日修复，全部落地）

三个问题类型当天修完（详见 Doc/design/low_pref_code_structure.md）：

| 类型 | 修法 | 修后 pref vs naive raw |
|---|---|---|
| max_acc | gen 换算常量 `ACTION_DELTA_TO_EE_ACC=6.0`（旧 0.0125 低估 ~500×，enforce 从未触发）+ AccRange `final_raw`=峰值 Δ‖ee_vel‖ | 1-3_107: 0.089 vs 0.165；1-3_109: 0.087 vs 0.183（重生成后步数 135→278 / 149→306，轨迹真的变缓） |
| avoid_object | AvoidToObject `final_raw`=**transport 段** min-dist（task_plan 对象 0.15m 门控排除任务必经区；ungated min-dist 兜底） | 1-4_034: 0.286 vs 0.205（gap 8.2cm；门控前 0.229 vs 0.205 仅 2.4cm——最近点在终点必经区，被正确剔除） |
| rotation | 重写 metric：raw=横移步（‖Δxy‖>1mm）平均 quat 旋转角，替换反语义的"descent 无旋转" | 1-8_114: 0.0104 vs 0.0035（~3×，方向对） |

顺带清扫：apply_low_pref.py 840→475 行（v0 函数×2、shadowed 重复 def、
resample_traj_constant_speed、rescale_traj_speed_safe 一族全 dead）+ motion_gen import 修剪。
60 单测过 + 清扫后 gen e2e smoke 通。
