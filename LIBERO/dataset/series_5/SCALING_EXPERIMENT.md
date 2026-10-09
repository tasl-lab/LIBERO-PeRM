# LIBERO-Scale — RQ3 Scaling 实验设置

**Slots**:
- **perm_0_1** (no-preference baseline) — `pick blue ketchup` simple pick-and-put，**只用于 baseline anchor**
- **perm_1_1** (motion: Speed up) — 主 scaling slot
- 其他 3 slot (2-5 / 3-2 / 4-3) 留 future work
**Model**: open-pi0.5（单 model, 单 seed）
**预算**: 9 ckpt（含 B0 no-pref baseline + M1-M5 scaling + L1/L3 layout-vs-data + L4 ∅-decoupling ablation）× ~12hr = ~108hr ≈ 4.5 天
**对应 experiment.md**: Phase 2a / H

---

## 7 Ckpt 配置（拆 M + L 两部分）

### Part B: No-preference baseline (1 ckpt)

| Cell | Slot | Task | Pref | (K, N) | Total | 用途 |
|---|---|---|---|---|---|---|
| **B0** | perm_0_1 | `Pick the blue ketchup and place it in the basket` | (空，无 pref) | (1, 50) | 50 | 零偏好基线 anchor — 给 Fig 1 提供 "纯 task-following" 起点；与 M1 (有 Speed up) 对比能区分"task fitting"和"pref fitting" |

physical path: `dataset/series_0/perm_0_1/libero_032/`（mission_id `0-1_032`）。BDDL 复用 LIBERO `mission_032/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl` 改造：drawer/cabinet → basket，akita_black_bowl → blue_ketchup，goal 改为 `(In blue_ketchup_1 basket_1_contain_region)`。

### Part M: 严格交替 K-N scaling (5 ckpt) — 5-tier joint

每 cell joint train 5 tier (0-1 no-pref + 1-1 motion + 2-5 skill + 3-2 profile + 4-3 multi)。
Total demo = 5 × K × N。

| Cell | (K, N) | Total demo (5×K×N) | 变化 |
|---|---|---|---|
| M1 | (1, 50) | 250 | start (= H_K1) |
| M2 | (3, 50) | 750 | ↑K |
| **M3** ← shared | **(3, 150)** | **2250** | ↑N |
| M4 | (7, 150) | 5250 | ↑K |
| M5 | (7, 300) | 10500 | ↑N |

跨度 250 → 10500 demos（42x）；K∈{1,3,7}, N∈{50,150,300}。

### Part L: Constant-budget K-scan (2 new + 1 shared)

| Cell | (K, N) | Total demo (5×K×N) | 用途 |
|---|---|---|---|
| L1 | (2, 225) | 2250 | 2 layouts × 225 demos × 5 tier (NEW) |
| L2 = M3 | (3, 150) | 2250 | 3 layouts × 150 demos × 5 tier (shared) |
| L3 | (5, 90) | 2250 | 5 layouts × 90 demos × 5 tier (NEW) |

L1, L2, L3 同 budget=2250，K∈{2,3,5} 三点。

### Part L4: ∅-decoupling ablation (1 new ckpt, K=3 fixed)

在 M3 (K=3) 基础上，**给 4 pref tier (1-1/2-5/3-2/4-3) 各加 K=3 个 no_pref baseline mission**（共 +12 mission），让模型学到每个 pref 是 task-agnostic lan token，从 layout/task 解耦。

| Cell | Mission 组成 | Mission 数 | (K, N) | Total demo | 用途 |
|---|---|---|---|---|---|
| **L4** | 5 tier × K=3 (原) + 4 pref tier × K=3 ∅ (新) | 15 + 12 = **27** | (3, 85) | 27 × 85 = **2295** ≈ 2250 | 测 ∅-data 是否帮助 disentangle pref from layout |

**与 M3 比较**：同 K=3、同 budget≈2250、同 layout 集；唯一变量 = 是否含 12 个 ∅ mission。L4 vs M3 的 SR/SA gap 即 "∅-decoupling 增益"。

**12 个新 ∅ mission**（每 pref tier 各取 3 layout）：
- 1-1 (Speed up) ∅: 3 mission（perm_1_1 内同 layout 清掉 pref）
- 2-5 (Pick left side) ∅: 3 mission
- 3-2 (Prefer red) ∅: 3 mission（直接复制 BDDL，改 goal state）
- 4-3 (multi) ∅: 3 mission

详见 `TODO_LAYOUT_MAGI.md` Part D。

---

## 3 张主图

| Fig | x 轴 | y 轴 | 用 cells | 答 |
|---|---|---|---|---|
| Fig 1 (Main × SR) | log(total demo) | Success Rate | M1-M5 | 数据量 → 任务完成饱和 |
| Fig 2 (Main × SA) | log(total demo) | Satisfaction (pref reward) | M1-M5 | 数据量 → 偏好遵循饱和 |
| Fig 3 (Layout-vs-Data) | K ∈ {1,3,5} | SR & SA | L1, L2/M3, L3 | 等 budget 下 layout 多样性的真实贡献 |
| Fig 4 (∅-decoupling) | {M3, L4} | SR & SA | L2/M3, L4 | ∅-data 是否帮 disentangle pref（同 budget） |

每图 in-domain 实线 + layout-OOD 虚线 = 2 条线（Fig 3 是 2 组 bar）。

---

## Metric

- **SR** = `bddl_done` 比例（per cell 50 ep 平均：35 in-domain + 15 OOD）
- **SA** (Satisfaction) = `LowPrefAccumulator.summary()` per-episode mean reward（用 `SpeedRangePreference` reward）

---

## 数据需求

每 train layout 最多需 300 demos（M5 cell）。Slot 1-1 当前 example：
- `series_1/perm_1_1/libero_004/` (existing)
- 需魔改另外 6 layout 凑 K_max=7 train（参考 `TODO_LAYOUT_MAGI.md` 流程）

7 train layout × 300 demos = **2100 demos** 总（M5 cell 上限）。Layout-OOD eval 用 3 held-out layout（不需要生成 demo）。

---

## 训练流水线（per ckpt）

1. 从 7 train layout 中随机/固定挑 K 个，每个抽 N 个 demo → LoRA finetune dataset
2. base ckpt = pi0.5 official (`gs://openpi-assets/checkpoints/pi05_base/params`)
3. LoRA recipe 与 SingPref `pi05_singpref_indomain_lora_v1` 一致（rank/alpha/lr/optimizer/step）
4. ckpt 选 = 训练 step 在数据量 / 时间 budget 内 best-on-validation

---

## Eval（per ckpt 50 ep）

| 用途 | Layout 集 | Trial / layout | 总 ep |
|---|---|---|---|
| In-domain (实线) | 7 train layout | 5 | 35 |
| Layout-OOD (虚线) | 3 held-out layout | 5 | 15 |

7 ckpt × 50 ep = **350 episodes** 总 eval（~30s/ep → ~3hr）。

---

## Caveat

M 部分的 K 与 N 同步扩展导致两效应混合（每 step 既加 K 也累计 demo）。L 部分给出 3 K 值 constant-budget K-scan 作为 "layout 多样性 vs demo 多样性" 的直接证据，但仍是单 budget 点（450）。Future work：扩 L 到多 budget level（如 budget=150 / 1050）需额外 4-6 ckpt。

---

## 与其他文档的衔接

- `REFERENCE.md` — 4 slot suite 设计骨架
- `TODO_LAYOUT_MAGI.md` — 9 个 layout 魔改流程
- `ProjectProfile/experiments/experiment.md` Phase 2a — 实验设计权威源
