# Series 6 (LIBERO-PrefConflict) — Mission 生成 TODO

因为我们要在**同一个 Layout** 上生成两种冲突的数据，所以需要给 Preference 2（P2）在 P1 的 Layout 上增加新的数据，并保持 P1 / P2 都是同一个 task。Preference-Task Conflict 同理：在原 task layout 上新增一份指向 conflict-class 的 task variant。

---

## Part A — pp P2 same-layout（4 个新 mission）

每对 pp 在 P1 mission 的 layout 上加一份 P2 lang/pref。BDDL 与 P1 共用，只改 `mission_info.yaml` 的 `*_pref_lan` / prefs 配置。

| pair | source example (P1) | P2 pref（写入 mission_info） |
|---|---|---|
| 1 speed | `1-1_004` (Speed up) | `low_pref_lan: Slow down` + slow SpeedRange block |
| 2 height | `1-7_104` (Move higher) | `low_pref_lan: Close to ground when moving` + low_path block |
| 3 pick LR | `2-13_001` (Pick from left edge) | `high_pref_lan: Pick from right edge` |
| 5 size | `3-5_120_1` (Prefer bigger) | `high_pref_lan: Prefer smaller objects`（复制 BDDL，改 goal state 大→小） |

> pp4 push/drag (`2-8_ood3` / `2-9_ood3`) 已 same-layout 存在，0 新生。

---

## Part B — pt NoPreferenceData（3 个新 mission）

给以下 3 个 source mission 去掉 preference，task lan 改成**点名 pref 不偏好的 class**（让 conflict 明显），生成 NoPreferenceData。

| pt | source mission | 原 pref（清空） | 新 task lan 点名 |
|---|---|---|---|
| pt1 color | `3-1_006_1` | Prefer green objects | pick the **blue/red** `<obj>` |
| pt3 category | `3-9_118_1` | Prefer dairy products | pick the **canned** `<obj>` |
| pt5 quantity | `3-8_007_1` | Serve double portion | serve **single / one** portion |
