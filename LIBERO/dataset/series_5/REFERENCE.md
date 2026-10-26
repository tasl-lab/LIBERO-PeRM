# LIBERO-Scale (Series 5) — Suite Reference

**Status**: G-stage — 4 pref locked, 1 example per slot 已就位（在原 series_X 路径下），9 layouts/slot pending magi
**Owner**: experiment.md / Phase 2a / RQ3 (sub-Q4 learning efficiency)
**Updated**: 2026-05-01

> **重要**：series_5/ 目录**不存任何 mission**——只有 REFERENCE.md + TODO_LAYOUT_MAGI.md 两份说明。所有 4 slot 的 mission 都直接放在原 series_X/perm_X_Y/ 路径下（与 mission_id 解析一致），无需特殊路径 redirect。

---

## 4 Slot 配置

按 P_Bench 3-tier preference 系统切分：

| Slot | Tier | Pref | 物理路径（实际 mission 在这） | Example (1 个) |
|---|---|---|---|---|
| **1-1** | motion (low-level) | `Speed up` | `dataset/series_1/perm_1_1/` | `libero_004/`（已存在，与 SingPref 共用） |
| **2-5** | skill (high-level) | `Pick from left side` | `dataset/series_2/perm_2_5/` | `libero_116/`（已存在，与 SingPref 共用） |
| **3-2** | profile (user-profile) | `Prefer red objects` | `dataset/series_3/perm_3_2/` | `libero_097_1/`（已存在，含 `no_pref_layout/`，与 SingPref 共用） |
| **4-3** | **multi** (motion + skill) | `Speed up` + `Pick from left side` | `dataset/series_4/perm_4_3/` | `libero_116/`（新增，由 2-5/116 改写：加 `low_pref_lan: Speed up` + `speed_range` low_prefs entry） |

**核心约束**（每 slot 内 10 个 mission）：
- ✅ task 不变（language / goal / obj_of_interest 完全相同）
- ✅ pref 不变
- 🔄 layout 不同（fixture / object 种类、数目、初始位置）

**复用机制**：1-1 / 2-5 / 3-2 的 example 直接复用 SingPref 现有 mission（同 mission_id），无需 copy。Scale suite 通过 `task_suites.yaml:liberoscale.missions` registry 圈出归属。

---

## Scale 参数

- 每 slot：7 train + 3 held-out = 10
- Train mission 总计 4 × 7 = **28** (≤ 30 ✓)
- Held-out OOD eval cell 总计 4 × 3 = **12**（仅 BDDL+mission_info.yaml，无 demo HDF5）
- 4 pref **混合训练单 ckpt**；按 (K, N) 双轴 scale (1,1)→(3,3)→(5,5)→(7,10)→(7,20)
- Test language **每条只挂单一 pref tier**（4-3 例外，挂 motion+skill 联合）

---

## G-stage TODO

详见 `TODO_LAYOUT_MAGI.md`。摘要：

- [ ] 1-1 / 2-5 / 3-2 / 4-3 各魔改 9 个 layout（共 36 个新 mission）
- [ ] Train 28 mission 试跑 SR ≥ 90%；held-out 12 仅写文件
- [ ] populate `LIBERO/configs/task_suites.yaml:liberoscale.missions`

---

## 与其他 suite 的边界

- **vs SingPref**：SingPref 用 38 唯一 layout，Scale 在每 slot 内堆 10 layout 同 task；mission_id 1-1_004 等共用
- **vs Compose**：Scale 4-3 直接训 (P1+P2) joint demo 测 multi-pref 数据效率；Compose 测 compose emergence
- **vs Conflict**：Scale 4 pref 互不冲突；Conflict 用对立 pref pair

## Cross-references

- 实验设计：`ProjectProfile/experiments/experiment.md` Phase 2a (G, H)
- Suite registry：`LIBERO/configs/task_suites.yaml:liberoscale`
- 魔改流程：`LIBERO/dataset/series_5/TODO_LAYOUT_MAGI.md`
