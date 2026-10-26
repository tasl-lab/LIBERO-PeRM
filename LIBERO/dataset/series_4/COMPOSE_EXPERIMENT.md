# LIBERO-Compose — Data Generation Spec

**4 layouts × 4 conditions = 16 missions**。每 layout 同 BDDL，pref 文本不同。

---

## 4_3 / libero_116 — `Put the ketchup in basket`

| Condition | mission_id | Pref text | Source / 状态 |
|---|---|---|---|
| nopref | `4-3_116_combo_nopref` | (空) | NEW |
| A | `4-3_116_combo_A` | `Pick from left side` (skill) | cp 2-5_116 ✓ |
| B | `4-3_116_combo_B` | `Speed up` (motion) | NEW |
| AB | `4-3_116_combo_AB` | `Pick from left side + Speed up` | cp 4-3_116 (Scale) ✓ |

---

## 4_4 / libero_001 — `Put the bowl in the bottom drawer of the cabinet and close it`

| Condition | mission_id | Pref text | Source / 状态 |
|---|---|---|---|
| nopref | `4-4_001_combo_nopref` | (空) | cp 3-2_001_1/no_pref_layout/ ✓ |
| A | `4-4_001_combo_A` | `Rotate when placing` (skill) | NEW |
| B | `4-4_001_combo_B` | `Prefer red objects` (UP) | cp 3-2_001_1 ✓ |
| AB | `4-4_001_combo_AB` | `Rotate when placing + Prefer red` | NEW |

---

## 4_5 / libero_118m — `Put the milk in basket`（BDDL: OJ→milk renamed）

| Condition | mission_id | Pref text | Source / 状态 |
|---|---|---|---|
| nopref | `4-5_118m_combo_nopref` | (空) | NEW |
| A | `4-5_118m_combo_A` | `Pick from top` (skill) | NEW |
| B | `4-5_118m_combo_B` | `Prefer dairy products` (UP) | NEW（基于 milk bddl 重跑）|
| AB | `4-5_118m_combo_AB` | `Pick from top + Prefer dairy` | NEW |

---

## 4_6 / libero_068 — `Pick up the ketchup and put it in the tray`

| Condition | mission_id | Pref text | Source / 状态 |
|---|---|---|---|
| nopref | `4-6_068_combo_nopref` | (空) | NEW |
| A | `4-6_068_combo_A` | `Lay object down when placing` (skill) | cp 2-14_068 ✓ |
| B | `4-6_068_combo_B` | `Serve double portion` (UP) | **NEW + BDDL 加 ketchup_2** |
| AB | `4-6_068_combo_AB` | `Lay down + Serve double portion` | **NEW + BDDL 加 ketchup_2** |

---

## 待生成 demo 清单（11 个 NEW）

```
4-3_116_combo_nopref
4-3_116_combo_B
4-4_001_combo_A
4-4_001_combo_AB
4-5_118m_combo_nopref
4-5_118m_combo_A
4-5_118m_combo_B
4-5_118m_combo_AB
4-6_068_combo_nopref
4-6_068_combo_B   ← bddl 改造后
4-6_068_combo_AB  ← bddl 改造后
```

预计 11 × 25 demo ≈ 275 demo，data_gen 单卡 ~6 hr。
