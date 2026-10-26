# Series 5 (LIBERO-Scale) — L4 ∅-decoupling Mission TODO

L4 cell 在 M3 (K=3) 基础上加 4 pref tier × 3 layout 的 no-preference baseline，让模型学到 pref 是 task-agnostic lan token，从 layout/task 解耦。

**新增 12 个 ∅ mission**（4 pref tier × 3 layout）。每个 ∅ mission：复制对应 source mission，**清掉 pref**（`high_pref_lan` / `low_pref_lan` / `*_prefs` 都置空），保留 task lan / goal；profile tier 例外，需复制 BDDL 后改 goal state（去除颜色/类别限定）。

| Tier | Source（3 layout，与 M3 同 layout） | 改 mission_info / BDDL |
|---|---|---|
| **1-1** Speed up | `1-1_004` + 另 2 个已 magi 的 layout | mission_info 清 `low_pref_lan` + `low_prefs: []`；BDDL 不变 |
| **2-5** Pick from left side | `2-5_116` + 另 2 个已 magi 的 layout | mission_info 清 `high_pref_lan` + `high_prefs: []`；BDDL 不变 |
| **3-2** Prefer red | `3-2_097_1` + 另 2 个已 magi 的 layout | 复制 BDDL 改 goal state（去颜色限定）；mission_info 清 pref |
| **4-3** multi (Speed up + Pick from left side) | `4-3_116` + 另 2 个已 magi 的 layout | mission_info 清 `high_pref_lan` + `low_pref_lan` + 两 prefs blocks；BDDL 不变 |
