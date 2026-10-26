# Series 6 — LIBERO-PrefConflict Pair Reference

**Suite**: LIBERO-PrefConflict (RQ5)  |  **Test mission**: **13**  |  **New mission_id**: 7 (4 pp P2 + 3 pt _c)  |  **R₂ aux**: 3 Task_B (reused)
**Stance**: task-first（见 `experiments/experiment.md` Decision 5）。pp 测 policy 如何处理对立 pref；pt 测 policy 是否守住 task。pt 训练加 (B, ∅) 解耦控制（让 P_B 与 Task_B disentangle）。

---

## 子 suite 划分

| 子 suite | conflict 数 | test mission | aux mission（不进 eval） |
|---|---|---|---|
| Pref-pref conflict (pp) | 5 pair | 5 × 2 = **10**（P2 与 P1 same-layout，可直接做 R₁ training data） | 0 |
| Task-pref conflict (pt) | 3 cell | 3 × 1 = **3** | +3 (R₂ Task_B，复用 SingPref/Series 3) |
| **总** | 8 | **13** | **+3** |

---

## 1. Pref-Pref Conflict (pp) — 5 pair × 2 = 10 mission

5 pair 全部 same-layout（P2 在 P1 layout 上挂对立 pref）。pref 文本即 mission_info `high_pref_lan` / `low_pref_lan`。

| # | pair-type | P1 mission | P1 pref | P2 mission | P2 pref | P2 状态 |
|---|---|---|---|---|---|---|
| 1 | M-M | `1-1_004` | Speed up | `1-2_004` | Slow down | NEW |
| 2 | M-M | `1-7_104` | Move higher | `1-5_104` | Close to ground when moving | NEW |
| 3 | S-S | `2-13_001` | Pick from left edge | `2-1_001` | Pick from right edge | NEW |
| 4 | S-S | `2-8_ood3` | Push instead of pick-and-put | `2-9_ood3` | Drag instead of pick-and-put | EXISTS (same BDDL) |
| 5 | U-U | `3-5_120_1` | Prefer bigger objects | `3-15_120_1` | Prefer smaller objects | NEW |

---

## 2. Task-Pref Conflict (pt) — 3 cell × 1 = 3 mission

每个 cell：**复用 Series 3 mission**（保留 original pref），**改 task lan** 指向 pref 不偏好的 object/位置。新 mission 后缀 `_c` ("conflict variant")。

| # | dim | source mission | new mission_id | original pref（保留） | 新 task lan |
|---|---|---|---|---|---|
| pt1 | color | `3-1_006_1` | `3-1_006_1c` | Prefer green objects | "pick up the **red** ..." |
| pt3 | category | `3-9_118_1` | `3-9_118_1c` | Prefer dairy products | "pick up the **canned** ..." |
| pt5 | quantity | `3-8_007_1` | `3-8_007_1c` | Serve double portion | "serve **only one** portion" |

---

## 3. (B, ∅) 解耦控制 — R₂ 训练辅助

每个 pt cell 加一条 `(Task_B, ∅)` 让模型学到 P_B 是 task-agnostic pref token（避免 P_B 被 entangle 进 Task_B）。

| pt | Task_B mission | Task_B 来源 | (B, P_B) | (B, ∅) |
|---|---|---|---|---|
| pt1 | `3-1_004_1` | SingPref pool | hdf5 已有 | `no_pref_layout/` 已有 |
| pt3 | `3-9_063_1` | Series 3 (non-SingPref) | demo 待补 | `no_pref_layout/` 待确认 |
| pt5 | `3-8_070_1` | Series 3 (non-SingPref) | demo 待补 | `no_pref_layout/` 待确认 |

R₂ 训练数据 = (Task_A_c, ∅) + (Task_B, P_B) + (Task_B, ∅)；测试 cell = (Task_A_c, original_pref)。

---

## 4. Mission Generation（最终版）

**Same-layout 设计**：每个 pp pair 在同 layout 上挂 P1 / P2 两条 mission；(Layout, ∅) baseline 复用该 layout source mission 的 no-pref 版本，**不单独造**。pt 同理：Task_A_c 有 BDDL/init，(Task_A_c, ∅) 复用同 BDDL 的 no_pref_layout/ 子目录。

| 类型 | 数量 | 复用 | 新生 mission_id | 新生 demo |
|---|---|---|---|---|
| pp P1 (5 pair) | 5 | 5 (SingPref + OOD) | 0 | 0 |
| pp P2 (5 pair) | 5 | 1 (pp5 已 same-layout) | **4** | 4 × 25 = 100 |
| pt Task_A_c (3 cell) | 3 | 0 | **3** | 3 × 25 = 75 |
| (Layout, ∅) for pp/pt | 8 | 8 (复用 source / no_pref_layout) | 0 | 0 |
| R₂ Task_B aux (3 cell) | 3 | 1 (SingPref) + 2 (Series 3，demo TBD) | 0 | 0~50（视 demo 已有性） |
| **总** | 24 | 17 | **7** (≈ 8 task) | **175** (+0~50) |

User 视角的"8 个 task" = 5 pp P2 + 3 pt Task_A_c（pp5 P2 物理已存但概念上算"P2 重新生成"）。

---

## Pending / Risk

- pt3, pt5 的 Task_B（`3-9_063_1` / `3-8_070_1`）是否已有 hdf5 demo —— 需扫 `VLA_data/SinglePref/` 与 Series 3 目录确认
- pt _c BDDL 的 :objects 必须含 conflict-class 物体（如 pt1 scene 必须有 red 变体；参 `Preference/pref_gen.py:129-199` 红物清单）
- 不再考虑 cross-type pair（M-S / M-U / S-U）；当前 5 pair 是 M-M ×2 + S-S ×2 + U-U ×1

---

## 关联文档

- `ProjectProfile/experiments/experiment.md` — Phase 2c (I 节点)
- `LIBERO/configs/task_suites.yaml:prefconflict` — registry
- `LIBERO/dataset/series_6/task_suite.yaml` — 结构化 pp_pairs / pt_cells / training_aux
- `LIBERO/dataset/series_6/TODO_DATA_GEN.md` — 8~10 个新生 mission 的工作清单
