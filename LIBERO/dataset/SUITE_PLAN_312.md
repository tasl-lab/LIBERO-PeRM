# 312 tasks：现有 setting 缺什么、怎么补（pref × layout 兼容矩阵版）

更新：2026-09-02。paper = `paper/1855_LIBERO_PeRM_Benchmarking_ (1).pdf`。
底表来源：`LIBERO/dataset/LIBERO_mission/`（125 个 base 场景）、`series_1..4` 的 153 个 mission 目录、`MotionGen/heuristic_skill/heuristic.py` + `Preference/` 的物体约束。

## 0. paper 里三个互相打架的总数

| 来源 | Single | Scale | Conflict | Compose | 合计 |
|---|---|---|---|---|---|
| 正文 / 摘要 / 贡献 / Fig 1（6 处） | 100 | 100 | **48** | 64 | **312** |
| Fig 2(a) 32.9 / 32.9 / 13.2 / 21.1 | 100 | 100 | **40** | 64 | **304** |
| Fig 2(b) 19.4 / 26.5 / 29.7 / 15.5 / 9.0 | motion 60 · skill 82 · profile 92 · multi 48 · ∅ 28 | | | | **310** |
| Fig 2(c) 14.5 / 25.5 / 27.3 / 32.7 | motion 8 · skill 14 · profile 15 · multi 18 | | | | 55 |

三张饼基数各异，是草案凑的。**锚正文 312 = 100/100/48/64，Fig 2 最后按实际表重画。** (c) 的 8/14/15 正好是现在的家数，说明 pref 家不用加，缺的是 layout 和搭配。
"task" 统一定义：一条 `(layout, task language, preference prompt)` = registry 里一个 mission_id + prompt 条件。

## 1. 现状

| suite | paper | 现有 | 备注 |
|---|---|---|---|
| SinglePref | 100 | 37 家 × 2 = 74 过审/待审；**mission 目录其实有 94 个** (motion 23 · skill 32 · profile 39) | 多出的 20 个：有的失败（2-6_042、1-7_039、3-13_046_1、3-5/3-13_120_1）、有的没配 naive（2-1_001、1-2_062、3-3_019_1、3-8_070_1 …） |
| PrefScale | 100 | 5 slot × 10–13 layout = 55 + 12 配对 ∅ + 12 pref-ood 格 | **raw hdf5 已删**，只剩 lerobot 归档 |
| PrefConflict | 48 | 10 pp + 3 pt = 13 | 16 个 hdf5 |
| PrefCompose | 64 | 4 layout × 4 = 16 | 4 个 eval-only 只有 5 条 |
| ComprePref | 不计 | 34 × 2 prompt | 跟 Single 走 |

## 2. 现有 setting 的"料"：10 个 layout 组

按场景 / 物体 / 固定装置把 125 个 base 场景归成 10 组（组内 layout 对 pref 的支持度一样）：

| 组 | 场景 | base ids | 关键物体 / 装置 |
|---|---|---|---|
| G1 客厅桌 + 篮子 | LIVING_ROOM_SCENE1/2 | 004–006, 056–064, ood4 | 罐/盒/瓶（soup, cheese, tomato, ketchup, OJ, milk, butter）+ basket |
| G2 客厅桌 + 托盘 | SCENE3 | 065–069 | 同上 + wooden_tray |
| G3 客厅桌 双碗 | SCENE4 | 070–074 | akita_bowl ×2, salad_dressing, pudding, tray |
| G4 客厅桌 杯 + 盘 | SCENE5/6 | 007, 075–082 | porcelain / red / white_yellow mug, plate ×2 |
| G5 厨房 三层柜 | KITCHEN_SCENE1/2/4/5/10 | 001, 010–027, 032–042 | bowl(×1–3), plate, wine_bottle+rack, ketchup, butter ×2, 3 层抽屉 |
| G6 GOAL 桌 | libero_goal | 100–110 | bowl, plate, cream_cheese, wine_bottle, 灶台, 酒架, 3 层抽屉 |
| G7 地面 + 篮子 | libero_object | 111–120, ood3 | 每场 6 种罐/盒/瓶 + basket |
| G8 书房 | STUDY_SCENE1–4 | 009, 083–099 | book(s), mug, desk_caddy 四格, 两层书架 |
| G9 灶台 | KITCHEN_SCENE3/8/9 | 000, 003, 028–031, 048–055 | moka_pot ×1–2, frypan, white_bowl, 灶台, 两层架 |
| G10 微波炉 | KITCHEN_SCENE6/7, ood1/2 | 002, 043–047 | mug ×2 / white_bowl+plate / blue_plate, microwave |

## 3. pref 家 × layout 组 兼容矩阵

✓ 原生可挂（物体 / 装置都在）· ○ 加一个孪生物体或改 bddl 才行 · – 不行。**现有** = 已有 mission 目录的 layout。**缺口** = 按"每家 ≥3 个不同组"看差什么。

物理约束来源：碗 / 盘 / 杯只能 edge 抓（plain pick 抬不起 rim）；`pick_handle` 只认 4 种 mug；top-angle / side / front 只认瓶罐盒；push / drag 只对 basket；lay_down 要细长物 + 宽平面；rotate-placing 要 caddy 格；avoid 要路径上有具名障碍；drawer 层级偏好要 3 层柜；profile 要场景里有属性不同的候选（recolor / resize 孪生资产都在 `available_obj_set` 里）。

| 家 | preference | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 | 现有 layout | 缺口 / 该去哪补 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | speed up | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | 004, 006 (G1) | 同组两个；补 G5/G7（034 / 114）跨组 |
| 1-2 | slow down | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | 062, 065, 113 | 够，三组齐 |
| 1-3 | acc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | 107, 109 (G6) | 同组；补 G1/G2（004 / 065） |
| 1-4 | avoid X | ○ | ○ | ○ | ○ | ✓ | ✓ | ○ | ○ | ✓ | ✓ | 034 (G5), 053 (G9) | 补 G6（109 避酒架 / 灶台） |
| 1-5 | close to ground | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ✓ | ✓ | – | – | 034, 070, 114 | 够 |
| 1-6 | straight | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | 067, 071, 105 | 够 |
| 1-7 | higher | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | 015, 104, 114 (039 失败) | 够 |
| 1-8 | rotate moving | ✓ | ✓ | ○ | ○ | ○ | ○ | ✓ | ○ | ✓ | – | 049, 061, 114 | 够 |
| 2-1 | right edge | ○ | ○ | ✓ | ✓ | ✓ | ✓ | – | – | ✓ | ✓ | 001, 019, 077 | 够（001 缺 naive 配对） |
| 2-2 | back edge | ○ | ○ | ✓ | ✓ | ✓ | ✓ | – | – | ✓ | ✓ | 019, 077 | 补 G3/G6（070 / 104） |
| 2-3 | handle | – | – | – | ✓ | – | – | – | ✓ | – | ✓ | 075, 077 (G4) | 只 3 组能挂；补 G8（086 白黄杯）或 G10（002） |
| 2-4 | top angle | ✓ | ✓ | – | – | ✓ | ○ | ✓ | – | – | – | 112, 119 (G7) | 同组；补 G1/G5（058 / 042） |
| 2-5 | left side | ✓ | ✓ | – | – | ✓ | ○ | ✓ | – | – | – | 112, 116 (+11 变体) | 够（scale slot） |
| 2-6 | right side | ✓ | ✓ | – | – | ✓ | ○ | ✓ | – | – | – | 119 (042 失败) | **只剩 1 个**；补 G1/G2（058 / 068） |
| 2-7 | horizontal | – | – | – | – | – | – | – | – | – | ✓ | ood1, ood2 | 天花板 2，不扩 |
| 2-8 | push | ○ | – | – | – | – | – | ✓ | – | – | – | ood3, ood4 | 天花板 2（可加 111–120 的"move basket"） |
| 2-9 | drag | ○ | – | – | – | – | – | ✓ | – | – | – | ood3, ood4 | 同上 |
| 2-10 | throw | ○ | ✓ | ✓ | – | ✓ | – | ○ | – | – | – | 015 (G5), 065 (G2) | 补 G3（071 pudding 扔进 tray） |
| 2-11 | front | ✓ | ✓ | – | – | ○ | ○ | ✓ | – | – | – | 062, 116, 118 | 够 |
| 2-12 | rotate placing | – | – | – | – | – | – | – | ✓ | – | – | 009, 085, 093 (G8) | 只 1 组，3 个都在 G8，够 |
| 2-13 | left edge | ○ | ○ | ✓ | ✓ | ✓ | ✓ | – | – | ✓ | ✓ | 001, 034 (G5) | 同组；补 G3/G4（070 / 077） |
| 2-14 | lay down | ○ | ✓ | – | – | ✓ | ✓ | ○ | – | – | – | 068 (G2), 109 (G6) | 补 G5（035 酒瓶放柜顶） |
| 3-1 | green | ✓ | ○ | ○ | – | ○ | ○ | ○ | – | – | – | 004, 006 (G1) | 同组；补 G7（115 green cheese） |
| 3-2 | red | ○ | ○ | ○ | ✓ | ✓ | ○ | ○ | ✓ | – | – | 001, 081, 097 | 够（scale slot） |
| 3-3 | yellow | ○ | ○ | ○ | ○ | ✓ | ○ | ✓ | ✓ | – | – | 019, 084, 117 | 够 |
| 3-4 | blue | ✓ | ○ | ○ | ✓ | ○ | ✓ | ○ | – | – | – | 058, 081, 104 | 够 |
| 3-5 | bigger | ✓ | ○ | ○ | ○ | ✓ | ○ | ✓ | – | – | – | 023, 064, 120 | 够 |
| 3-6 | upper drawer | – | – | – | – | ✓ | ✓ | – | – | – | – | 012, 036, 101 | 够 |
| 3-7 | middle drawer | – | – | – | – | ✓ | ✓ | – | – | – | – | 018, 101 | 只 2 组；补 G5（039 碗进中层） |
| 3-8 | double portion | ○ | ○ | ✓ | ✓ | ✓ | – | ○ | – | ✓ | – | 007, 070, 081 | 够 |
| 3-9 | dairy | ✓ | ○ | – | – | – | – | ✓ | – | – | – | 063, 118 | 只 2 组；补 G1 另一层（062）或 G7（117） |
| 3-10 | sweet drink | ✓ | ○ | – | – | – | – | ✓ | – | – | – | 062, 117 | 同上；补 063 / 115 |
| 3-11 | sour sauce | ○ | ○ | ✓ | – | – | – | ✓ | – | – | – | 072, 112 | 补 G7（116 / 119） |
| 3-12 | boxed | ✓ | ✓ | – | – | ○ | ✓ | ✓ | – | – | – | 036, 060, 109 | 够 |
| 3-13 | bottled | ✓ | ✓ | – | – | ○ | ✓ | ✓ | – | – | ✓ | 046, 120 | 补 G1/G2（060 / 068） |
| 3-14 | canned | ✓ | ✓ | – | – | ○ | ✓ | ✓ | – | – | – | 107, 114 | 补 G1（060） |
| 3-15 | smaller | ✓ | ○ | ○ | ○ | ✓ | ✓ | ✓ | – | – | – | 040, 041, 106, 120 | 够 |

读法：
- **motion 8 家哪都能挂**，缺的只是"跨组"（1-1 / 1-3 现在两个都在同一组，layout 泛化没料）。
- **skill 里 2-7 / 2-8 / 2-9 / 2-12 / 2-3 是硬天花板**（微波炉、篮子、caddy、mug 白名单），这 5 家维持 2–3 个，别硬扩；其余 9 家都有 ≥4 个组可去。
- **profile 里 3-6 / 3-7 只能在三层柜（G5/G6），3-9 / 3-10 / 3-11 只能在有对应饮料 / 酱的 G1/G7**，其余 10 家靠 recolor / resize 孪生几乎哪都能挂。

## 4. 哪些 layout 天然能承多种 pref（compose / conflict / scale 的料）

从 153 个现有 mission 目录里统计，**已经同时挂 ≥3 家** 的 layout：001、019、034、062、070、077、081、109、112、114、116、118、120；挂 2 家的：004、006、015、036、053、065、068、101、104、107、117、119。

| 用途 | 判据 | 现成的 layout（pref 对） |
|---|---|---|
| **Conflict pp**（同 layout、互斥） | 矩阵里同格都 ✓ 且物理上二选一 | 004 speed/slow · 114 ground/higher · 001 left/right edge · 077 handle/edge · 120 bigger/smaller · 101 upper/middle drawer · 117 yellow-milk/sweet-OJ · 081 red/blue · 065 slow/throw · 118 dairy/sweet · 119 top-angle/right-side · 062 slow/speed |
| **Compose**（同 layout、正交） | 两个 pref 作用在不同阶段 / 不同维度（motion × skill × profile） | 062 slow+front+sweet · 109 acc+lay-down+boxed · 112 top-angle+sour · 114 ground+canned · 034 avoid+left-edge · 070 ground+double · 019 right-edge+yellow · 104 higher+blue · 107 acc+canned · 015 higher+throw · 081 blue+double · 053 avoid+straight |
| **Scale**（一个 pref 跨最多组） | 矩阵一整行 ✓ 最多 | 1-1 speed（10 组）、3-2 red（8 组，靠孪生）、2-5 left side（4 组：G1/G2/G5/G7）、4-3 = 2-5 ∩ 1-1 |

## 5. 按矩阵凑 312

### 5.1 SinglePref 74 → 100：规则 = "每家 ≥2，能跨 ≥3 组的家补到 3"

- 26 家补第 3 个（motion 8 家全补；skill 9 家：2-1、2-2、2-4、2-5、2-6、2-10、2-11、2-13、2-14；profile 9 家：3-1、3-3、3-4、3-5、3-8、3-12、3-13、3-14、3-15）→ 78 + 22 = 100 … 实际按目录：现有 94 个目录里能救回来的先用（2-1_001、1-2_062、3-3_019_1、3-4_081_1、3-6_036_1、3-8_070_1、3-12_036_2、3-15_041_1、1-5_034、1-6_067、1-8_049、2-12_009 共 12 个已有 dir/SR），再按矩阵"缺口"列新挂 14 个。
- 11 家留 2：2-3、2-7、2-8、2-9、2-12（硬天花板）+ 3-6、3-7、3-9、3-10、3-11（组受限）+ 1-2 已 4 个抵一个。
- 结果 motion 24 · skill 35 · profile 41 = 100（逐条见 5.1.1）。第三个一律选**不同组**的 layout（给 Scale / 泛化留 layout 多样性）。

#### 5.1.1 Single 100 逐条（2026-09-03 定稿，按 S1/S2/S3 实际过审的 layout 算）

**A. 已在审的 74**（S1 motion 16 · S2 skill 28 · S3 profile 30，每个 10 pref + 10 naive）。registry `perm_missions.yaml` 的 pairings 有 8 处跟实际审的对不上，2026-09-03 已同步：1-2 062→113 · 1-6 067→105 · 1-7 039→104 · 1-8 049→114 · 2-6 042 删掉（116_1 是 2-5 的抽屉变体，记在 2-5 下作站位）· 3-2 086_1→081_1 · 3-3 087_1→084_1 · 3-13 046_1/120_1→046_2/120_2。同步后 pairings = 88（74 + B 的 14）。
74 里 bulk 100 条已有 43 个（motion 16 全有、1-4 两个只有 50；skill 12；profile 15），**缺 bulk 的 31 个**：2-3_075 2-4_112 2-5_112 2-6_116_1 2-7_ood1/2 2-8_ood3/4 2-9_ood3/4 2-10_015 2-11_062 2-12_085/093 2-13_034 2-14_068 · 3-2_081_1 3-3_084_1 3-4_058_1 3-5_023_1/064_1 3-6_012_1 3-7_018_1 3-8_081_2 3-9_063_1 3-10_062_1 3-11_072_1 3-12_060_1 3-13_046_2/120_2 3-15_040_1。

**B. 现成目录直接升第 3 个的 14 个**（bddl + plan + 5–6 条 probe 视频都在，只差 10+10 审阅对 + bulk）。第 3 个的组都跟前两个不同：

| id | 组 | 前两个 | 已有 |
|---|---|---|---|
| 1-2_062 | G1 | 065 G2 / 113 G7 | probe 3 条 + 2 fail |
| 1-5_034 | G5 | 070 G3 / 114 G7 | probe 6 |
| 1-6_067 | G2 | 071 G3 / 105 G6 | probe 6，SR 1/1 |
| 1-7_114 | G7 | 015 G5 / 104 G6 | **bulk 100**（PrefConflict 下） |
| ~~1-8_049~~ → **1-8_071**（新挂，见 C） | G3 | 061 G1 / 114 G7 | 049 关炉子的 naive 只成 5/30，换成转着搬布丁进 tray |
| 2-12_009 | G8 | 085 / 093 G8（硬天花板家，允许同组） | **bulk 100** |
| 3-2_097_1 | G8 | 001 G5 / 081 G4 | probe 5 |
| 3-3_019_1 | G5 | 084 G8 / 117 G7 | probe 5 |
| 3-4_081_1 | G4 | 058 G1 / 104 G6 | probe 6 |
| 3-5_120_1 | G7 | 023 G5 / 064 G1 | **bulk 100** |
| 3-6_036_1 | G5 | 012 G5 / 101 G6（只有 G5/G6，允许） | probe 5 |
| 3-8_070_1 | G3 | 007 / 081 G4 | probe 4 |
| 3-12_036_2 | G5 | 060 G1 / 109 G6 | probe 4 |
| 3-15_120_1 | G7 | 040 G5 / 106 G6 | **bulk 100**（PrefConflict 下） |

备胎不用：2-1_001（跟 019 同 G5）、2-11_116（跟 118 同 G7）、3-15_041_1（跟 040 同 G5）、3-13_046_1/120_1（被 _2 替代）。2-5 在 Single 留 2（Scale slot 已经 12 个 layout）。

**C. 要新挂的 13 个**（12 + 替换 1-8_049 的 1-8_107；layout 都是已有 mission 的场景，只新写 pref 那层 + plan，脚手架 `scripts/scaffold_single_new12.py`）：

| ~~1-8_071~~ → **1-8_107** | G6 | 061 G1 / 114 G7 | 071 转着抓布丁放 tray 时掉在外面（90° / −45° 都 0/60）；改成 107 转着搬奶酪盒进碗 | 低 |

| id | 组 | 前两个 | 怎么挂 | 风险 |
|---|---|---|---|---|
| 1-1_034 | G5 | 004 / 006 G1 | 1-5_034 同 bddl，speed 是 scene-agnostic | 低 |
| 1-3_065 | G2 | 107 / 109 G6 | 1-2_065 同 bddl，换 acc 指标 | 低 |
| ~~1-4_109~~ → **1-4_070** | G3 | 034 G5 / 053 G9 | 109 上避碗 0.45 / 避盘 0.3 都 0/60（GOAL 柜子场景绕不开）；改成 070 搬碗进 tray 时避开另一只碗 0.3 m | **中** |
| 2-2_070 | G3 | 019 G5 / 077 G4 | 1-5_070 场景，碗 back-edge 抓 | 低 |
| ~~2-4_058~~ → **2-1_001**（现成，Conflict 也要它） | G5 | 2-4 番茄酱 top-angle 没录过 grasp 参数 0/60，2-4 留 2；名额给 2-1_001（同组，但 bulk 100 已在 PrefConflict） | 低 |
| 2-6_068 | G2 | 119 G7（116_1 其实是 2-5 的 left-side 抽屉变体，只是审阅时替 2-6 站位；registry 里记在 2-5 名下） | 2-14_068 场景，瓶子 right-side 抓 | 低 |
| 2-10_071 | G3 | 015 G5 / 065 G2 | 1-6_071 场景，pudding 扔进 tray | 低 |
| ~~2-13_077~~ → **2-13_070** | G3 | 001 / 034 G5 | 杯子 right-edge 放不上盘子 0/60；改成碗 right-edge 进 tray | 低 |
| ~~2-14_036~~ → **2-14_072** | G3 | 068 G2 / 109 G6 | 酒瓶横在抽屉沿上 0/60；改成沙拉酱放倒进 tray（072） | 低 |
| 3-1_115 | G7 | 004 / 006 G1 | 地面篮子加 green cheese 孪生 | 低 |
| 3-13_060_2 | G1 | 046 G10 / 120 G7 | 3-12_060_1 场景，候选改成瓶 vs 盒/罐 | 低 |
| 3-14_060_3 | G1 | 107 G6 / 114 G7 | 同上，罐 vs 瓶/盒（060 三兄弟顺便给 Conflict pt 用） | 低 |

**D. 分批**：batch 1 = B+C 共 26 个的 10 pref + 10 naive（520 条，进 S1/S2/S3 审阅）；batch 2 = 过审后 bulk 100（A 缺 31 + B/C 缺 22 = 53 个 ≈ 5.3K，外加 1-4 两个补 50）。类型：motion 24 · skill 35 · profile 41。

### 5.2 PrefScale → 100：slot 不变，layout 往"跨组"长

| 组成 | 数 | 现状 → 要做 |
|---|---|---|
| 0-1 ∅ (032 家族) | 13 | 目录齐 |
| 1-1 speed 16 layout | 16 | 有 10（全在 G1 004 变体）→ +6 **换组**：065(G2) 070(G3) 077(G4) 034(G5) 104(G6) 114(G7) |
| 2-5 left side 16 | 16 | 有 12（floor/kitchen/table 三种几何）→ +4：058(G1) 068(G2) 042(G5) 111(G7) |
| 3-2 red 15 | 15 | 有 10（097 书架 + 篮子）→ +5：081(G4 原生红杯) 001(G5 红碗) 019 062 116 |
| 4-3 left+speed 16 | 16 | 复用 2-5 的 bddl 加 speed，+6 不用新 bddl |
| 4 pref slot 各 3 个配对 ∅ | 12 | 目录齐 |
| pref-ood 格 0-1_032_7/8/9 × 4 pref | 12 | prompt 变体 |
| **合计** | **100** | 13+16+15+16+16+12+12 = 100；新 bddl 15 个 + 4-3 复用 6 个 = 21 个新 mission；**88 个有 demo 的 mission 全部重生成**（pref-ood 12 个只是 prompt 变体，50 条/mission ≈ 4.4K） |

### 5.3 PrefConflict 13 → 48 = 6 种 pp 对 × 2 layout × 3 + 3 种 pt 格 × 2 layout × 2（2026-09-03 定稿，每种 setting 两个 bddl）

pp 每个 layout 出 3 个 task：P1 单 pref、P2 单 pref、P1+P2 联合 prompt。6 种对立 pref 对，每种两个 layout：

| pair-type | layout 1（都现成） | layout 2 | 新挂 |
|---|---|---|---|
| speed up ↔ slow down | 004（1-1 / 1-2） | 062（1-2_062 ✓ + **1-1_062**） | 1 |
| ground ↔ higher | 114（1-5 / 1-7） | 104（1-7_104 ✓ + **1-5_104**，height −0.12） | 1 |
| left edge ↔ right edge | 001（2-13 / 2-1） | 034（2-13_034 ✓ + **2-1_034**） | 1 |
| push ↔ drag | ood3（2-8 / 2-9） | ood4（2-8 / 2-9 都有） | 0 |
| bigger ↔ smaller | 120_1（3-5 / 3-15） | 023_1（3-5_023_1 ✓ + **3-15_023_1**，碗放小盘 vs 大盘） | 1 |
| handle ↔ edge | 077（2-3 / 2-1） | 075（2-3_075 ✓ + **2-1_075**，红杯抓边） | 1 |

pt 每个 layout 出 2 个 task：task 点名 pref 不选的那个 + 原 pref（冲突），和同一 task + ∅。3 种格，每种两个 layout：color 006_1c ✓ / **004_1c**（点名白色奶酪盒）、category 118_1c ✓ / **063_1c**（点名橙汁）、quantity 007_1c ✓ / **070_1c**（只放一只碗）。_c 的 demo 跟 task 走，不需要 naive 配对，只看 SR。

新挂 8 个 mission（5 pp P2 + 3 pt _c），全部走 batch 2；registry 已重写成 12 pp pair × 6 pt cell（`VLA_data/PrefConflict/tasksuite.yaml` 与 `series_6/task_suite.yaml` 同源）。现成 3 个 _c 的 mission_info 里 pref 文本原来写错（006_1c 写成 red、007_1c 空、118_1c 写成 sweet），已改回原 pref。类型：motion 8 · skill 12 · profile 10（size 4 + pt 冲突 prompt 6）· multi 12 · ∅ 6。

### 5.4 PrefCompose 16 → 64 = 8 种 A+B setting × 2 layout × 4 condition（2026-09-03 定稿）

A/B 单 pref 直接复用 Single 的 mission（同 layout 同 demo），每个 layout 只新做 ∅（naive plan）和 A+B 联合两个 mission：

| setting | 类型 | layout 1 | layout 2 | 新单 pref |
|---|---|---|---|---|
| 4-3 speed + left side | M+S | 116（现成 4 condition） | 112（2-5_112 ✓ + **1-1_112**） | 1 |
| 4-7 slow + front | M+S | 062（1-2 / 2-11 ✓） | 118（2-11_118 ✓ + **1-2_118**） | 1 |
| 4-8 avoid + left edge | M+S | 034（1-4 / 2-13 ✓） | 001（2-13_001 ✓ + **1-4_001**） | 1 |
| 4-9 higher + throw | M+S | 015（1-7 / 2-10 ✓） | 065（2-10_065 ✓ + **1-7_065**） | 1 |
| 4-10 ground + double | M+U | 070（1-5 / 3-8 ✓） | 007（3-8_007_1 ✓ + **1-5_007**） | 1 |
| 4-11 higher + blue | M+U | 104（1-7 / 3-4 ✓） | 058（3-4_058_1 ✓ + **1-7_058**） | 1 |
| 4-12 acc + canned | M+U | 107（1-3 / 3-14 ✓） | 114（3-14_114_1 ✓ + **1-3_114**） | 1 |
| 4-13 back edge + yellow | S+U | 019（2-2 / 3-3 ✓） | 077（2-2_077 ✓ + **3-3_077_1**，黄杯不用孪生） | 1 |

新 mission：8 个单 pref + 15 个 ∅ + 15 个 A+B（116 的 4 个现成），全部由 `scripts/scaffold_single_new12.py --compose` 生成；registry `VLA_data/PrefCompose/tasksuite.yaml` 重写。旧的 4-4 001（rotate+red）、4-5 118m（top+dairy）、4-6 068（lay-down+double）第二个 layout 都要造孪生或没有对应 grasp 参数，数据留着不计入。类型：motion 14 · skill 10 · profile 8 · multi 16 · ∅ 16。

### 5.5 汇总

| | Single | Scale | Conflict | Compose | 合计 | Fig 2(b) 现值 |
|---|---|---|---|---|---|---|
| motion | 24 | 19 | 8 | 14 | **65** | 60 |
| skill | 35 | 19 | 12 | 10 | **76** | 82 |
| profile | 41 | 18 | 10 | 8 | **77** | 92 |
| multi | 0 | 19 | 12 | 16 | **47** | 48 |
| no-pref | 0 | 25 | 6 | 16 | **47** | 28 |
| **合计** | **100** | **100** | **48** | **64** | **312** | 310 |

Fig 2(c)：motion 8 · skill 14 · profile 15 · multi 16（compose 组合）。∅ 47 vs 28 的差全在 Scale 的 ∅ slot + 配对 ∅，要贴回去得再多 9 个 pref layout，不建议。

## 6. 工作量 [INFERRED, MED]

| 项 | 新 mission / bddl | 新 demo | 瓶颈 |
|---|---|---|---|
| Single +26 & 补到 100/task | 12 新挂 + 14 现成 | ≈ 0.5K 审阅对 + 5.4K bulk | 每个要 pref 10/10 + naive 配对 + 过审 |
| Scale | 21 新 mission（15 新 bddl）；88 mission 重生成 | ≈ 4.4K | layout 魔改人工 |
| Conflict | 9 新 mission | ≈ 0.9K | 无（都是现成 pref 挂现成 layout） |
| Compose | 12 新 A+B | ≈ 1.6K | 联合 plan 逐个调 |
| **合计** | ≈ 54 新 mission（其中 15 个新 bddl）| **≈ 15K 条** | 机时 40–60 h（300–400 条/h），plan 调试 1–2 周 |

paper 里 "100+ trajectories per task, 40.8K rollout episodes" 也要一起改：补完 ≈ 31K 条，40.8K 只能把生成失败 rollout / naive 配对算进去。

## 7. 顺序

1. Single +26（流水线现成；ComprePref 顺手扩到 100 × 2）。
2. Compose 12 个 A+B（只差联合 demo）。
3. Conflict 9 个新 mission。
4. Scale：15 个新 bddl 人工过一遍，再批量重生成 88 个。
5. 重画 Fig 2、改 40.8K 那句。
