# Prompt 草案 — SinglePrefDataTest

## 格式

```
Task: <精简 task>. Preference: <pref>.
```

- `Task:` / `Preference:` 是 anchor，每帧训练 prompt 一致出现，让 model 把 pref 段独立学
- 句尾统一加 `.`（pref 段也是）
- 全 lowercase？目前 LIBERO 原文是 sentence case；保留 sentence case，避免引入额外变换

## 18 条草案

| mid | 原 task | 精简 task | pref |
|---|---|---|---|
| 1-1_004 | Put both the alphabet soup and the cream cheese box in the basket | Put alphabet soup and cream cheese in basket | Speed up by 2 times |
| 1-1_006 | Put both the cream cheese box and the butter in the basket | Put cream cheese and butter in basket | Speed up by 2 times |
| 1-4_034 | Put the black bowl in the bottom drawer of the cabinet | Put black bowl in bottom drawer | Keep away from wine bottle |
| 1-4_053 | Put the white bowl on top of the cabinet | Put white bowl on top of cabinet | Keep away from stove |
| 1-7_039 | Put the black bowl in the top drawer of the cabinet | Put black bowl in top drawer | Move higher |
| 1-7_015 | Put the chocolate pudding in the top drawer of the cabinet and close it | Put chocolate pudding in top drawer and close it | Move higher |
| 2-1_019 | Put the black bowl on the plate | Put black bowl on plate | Pick from right edge |
| 2-1_077 | Put the white mug on the left plate | Put white mug on left plate | Pick from right edge |
| 2-5_112 | Pick the BBQ sauce and place it in the basket | Put BBQ sauce in basket | Pick from left side |
| 2-5_116 | Pick the ketchup and place it in the basket | Put ketchup in basket | Pick from left side |
| 2-12_085 | Pick up the book and place it in the right compartment of the caddy | Put book in right caddy compartment | Put with rotation |
| 2-12_093 | Pick up the book and place it in the right compartment of the caddy | Put book in right caddy compartment | Put with rotation |
| 3-1_004_1 | Put both the alphabet soup and the cream cheese box in the basket | Put alphabet soup and cream cheese in basket | Prefer green objects |
| 3-1_006_1 | Put both the cream cheese box and the butter in the basket | Put cream cheese and butter in basket | Prefer green objects |
| 3-4_058_1 | Pick up the ketchup and put it in the basket | Put ketchup in basket | Prefer blue objects |
| 3-4_081_1 | Put the red mug on a plate | Put red mug on plate | Prefer blue objects |
| 3-8_007_1 | Put the white mug on the left plate and put the yellow and white mug on the right plate | Put white mug on left plate, yellow-white mug on right plate | Serve double portion |
| 3-8_081_2 | Put the red mug on a plate | Put red mug on plate | Serve double portion |

## 拼接示例

```
1-1_004 → "Task: Put alphabet soup and cream cheese in basket. Preference: Speed up by 2 times."
2-1_019 → "Task: Put black bowl on plate. Preference: Pick from right edge."
3-8_007_1 → "Task: Put white mug on left plate, yellow-white mug on right plate. Preference: Serve double portion."
```

## Review checklist

- [ ] task 精简后语义无歧义？（特别是 1-1_004 / 1-1_006 用了 "and"，原文是 "both ... and ..."）
- [ ] pref 文本要不要进一步改？（pref 段 paraphrase 你之前说要"直接、生动、简短"）
- [ ] anchor 词 `Task:` / `Preference:` 是否要换成更短的（如 `T:` / `P:` 或 `Goal:` / `Style:`）