# LIBERO-Compose (Series 4) — Suite Reference

**Status**: skeleton — 4 个 layout × 4 task per layout = 16 mission（perm_4_3 ~ 4_6 + 继承 4_1/4_2 待评估）
**Owner**: experiment.md / Phase 2b / RQ4 (a/b — compose / decompose only)
**Updated**: 2026-04-28

> 设计源：`ProjectProfile/experiments/series_4_design.md`
> Pref Gen / Layout Gen 不在本 suite — 已被 Scale (series_5) 覆盖。

---

## 结构总览

| 项 | 数量 |
|---|---:|
| Layout (combo) | 4 |
| Task per layout (condition) | 4：`nopref / A / B / A+B` |
| **Total mission** | **16** |
| 新 demo set | **4**（仅 A+B 联合）|
| 复用 SingPref + LIBERO 基线 demo | 12 (nopref / A / B) |
| Total demo (≈ 25/set) | ≈ 100 |

---

## 4 个 Layout（perm_4_3 ~ perm_4_6）

每个 perm_4_X 是 1 个 layout（共享 1 个 bddl），下面挂 4 个 mission_id：

```
perm_4_X/
├── libero_NNN_nopref/   # condition 1: 无 pref（base task）
├── libero_NNN_A/        # condition 2: + Pref A
├── libero_NNN_B/        # condition 3: + Pref B
└── libero_NNN_AB/       # condition 4: + Pref A+B（联合）
```

| pbench | layout (NNN) | Pref-type pair | Pref A | Pref B |
|---|---|---|---|---|
| **4_3** | 116 | skill + motion | `Pick from left side` (← 2-5_116) | `Speed up` (← 1-1, via Scale 4-3) |
| **4_4** | 001 | skill + UP | `Rotate when placing` (NEW on 001) | `Prefer red` (← 3-2_001_1) |
| **4_5** | 118m (milk-renamed) | skill + UP | `Pick from top` (NEW) | `Prefer dairy` (← 3-9_118_1, OJ→milk bddl) |
| **4_6** | 068 | skill + UP | `Lay object down` (← 2-14_068) | `Serve double portion` (NEW, needs ketchup_2 in bddl) |

> Layouts finalized 2026-04-28. Mission_id 后缀 `_combo_<nopref/A/B/AB>` 区分 4 condition。

---

## Demo 来源

| Condition | 来源 | 数量 (4 layout) |
|---|---|---:|
| `_nopref` | LIBERO 原始 demo（`libero_rlds_extracted` 或 `no_pref_layout/` 基线） | 4（复用） |
| `_A` | SingPref pool 同 layout + Pref A 的 mission（symlink HDF5） | 4（复用） |
| `_B` | SingPref pool 同 layout + Pref B 的 mission（symlink HDF5） | 4（复用） |
| `_AB` | **新跑 data_gen**（语言= `<task> + <Pref A> + <Pref B>`） | **4 新** |

**新生成 demo ≈ 4 × 25 = 100 demo**（GPU < 半天）。

---

## ckpt 与 eval（≤ 5 ckpt 上限）

| Ckpt | Train (per layout) | Test (per layout) | RQ | 优先级 |
|---|---|---|---|---|
| **C1 Decomp** | nopref + A + A+B | B（4 mission） | RQ4 b | **必跑** |
| **C2 Compose** | nopref + A + B | A+B（4 mission） | RQ4 a | **必跑** |
| **C3 Baseline** | SingPref-LoRA 既有 ckpt（无新训）| 全 16 mission（4 condition × 4 layout） | reference | optional（看时间）|

C1/C2 × 4 condition × 4 layout = 32 eval cell（必跑）；+ C3 = 48 cell。**Metric**：SR (%) + SA (preference satisfaction %)。

---

## 继承 (perm_4_1, 4_2)

来自 m1_4 / m1_5 重命名（commit `c6df1e7`）：

| pbench | libero_* | 原 pref |
|---|---|---|
| **4_1** | libero_034, libero_053 | Keep away from wine bottle / stove |
| **4_2** | libero_034, libero_070, libero_109_deprecated | Close to ground when moving |

**待决策**：是否改造为符合 4-condition 模板（替换/扩展），或保留为独立 motion-only 测试。如不动，从 16 mission 主表中独立。

---

## task_suites.yaml 注册（待更新）

```yaml
compose:
  description: RQ4 Combine — 4 layouts × {nopref, A, B, A+B}
  missions:
    - 4-3_116_nopref
    - 4-3_116_A
    - 4-3_116_B
    - 4-3_116_AB
    - 4-4_015_nopref
    # ... 共 16 条
```

> mission_id 格式：`{series}-{task}_{layout}_{condition}`，需 `parse_mission_id` regex 兼容（条件后缀字符串）。

---

## TODO

- [ ] User 确认 4 个 layout / Pref pair 选择（默认值是否 OK）
- [ ] 决策 4_1/4_2 归属
- [ ] Scaffold 16 个 `perm_4_X/libero_NNN_<cond>/mission_info.yaml`
- [ ] 同 layout 的 4 个 condition 共享 1 个 `*.bddl`，差别只在 mission_info 的 pref 字段
- [ ] 生成 4 个 A+B 联合 demo set（≈100 demo）
- [ ] Symlink nopref / A / B demo from SingPref + LIBERO baseline
- [ ] 注册 `compose` suite 到 `LIBERO/configs/task_suites.yaml`
- [ ] `parse_mission_id` 增加对 `_<condition>` 后缀的解析（或者改用 condition= eval 参数）
