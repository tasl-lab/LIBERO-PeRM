# Series 7 — LIBERO-Comprehensive: Paraphrasing Rules

Authoritative spec for how language variants under series 7 are generated.
Future generation passes (and any auto-LLM scripts) MUST follow these two
rules verbatim. Strings live in `LIBERO/dataset/series_7/language.yaml`.

Anchor that both rules paraphrase **away from**: `pref_canonical` in the
mission's `mission_info.yaml` (= the L1 / canonical pref string used by
SingPref). Series 7 has 2 settings — S1 (Rule 1) and S2 (Rule 2). There
are no L0 / L1 outputs under series 7; L0 (no-pref) and L1 (canonical)
are already covered by SingPref evaluation.

---

## CROSS-CUTTING INVARIANT (read first)

Both S1 and S2 MUST share an identical **action skeleton**:

```
<pick verb> <concrete target object name> [from <where>] <place verb> <destination>.
```

- The **target object name** must be the *concrete* object actually
  picked by the mission's `task_plan` — not a generic noun
  ("a drink", "a bowl", "a sauce"). For series 3, that means the
  pref-resolved object (e.g. `milk` for prefer_dairy, `red bowl` for
  prefer_red, `green cream cheese box` for prefer_green).
- The **destination** must be named (basket / bowl / blue plate / left
  plate / bottom drawer / top of cabinet / etc.).
- The **verb pair** (pick + put) must be present, even if fused into a
  single composite verb (load / transfer / move / pack / set).

The S1 vs S2 difference is **only in how the preference is conveyed**,
NOT in which object is acted on:

- S1 weaves the pref into the action description (manner adverb,
  attribute adjective, choice of synonym verb).
- S2 keeps the action description plain and adds a context fact that
  forces commonsense inference of the pref.

This invariant is what isolates the paraphrase axis: holding task
skeleton fixed, we vary only the pref-expression style. A prompt that
hides the target object would also test object-grounding, which is a
different RQ.

---

## Rule 1 — Fused paraphrasing (S1, no reasoning)

The instruction must be **one fluent sentence (or two tightly coupled
sentences) that fuses the task and the pref**. The pref is *not* a
separate appended clause — it is woven into how the task is described.
A reader recovers the canonical pref by reading the verb / adverb /
adjective choices alone, with no world-model lookup.

### What qualifies

| Mode                         | Example pref          | Example S1 (fused)                                                      |
|------------------------------|-----------------------|-------------------------------------------------------------------------|
| Adverb-fused                 | "Speed up"            | "Quickly load both items into the basket."                              |
| Verb-substitution            | "Throw instead of placing" | "Toss the alphabet soup into the tray."                            |
| Body-of-the-action fused     | "Pick by handle"      | "Grip the white mug by its handle and move it to the left plate."       |
| Adjective-into-object        | "Prefer green"        | "Pack the green cream cheese box into the basket."                      |
| Manner-clause fused          | "Close to ground"     | "Slide the pudding into the basket while keeping the arm low throughout." |

### Hard constraints

1. **No 'task. preference.' append form.** If you can split the prompt
   at a period and recover the canonical SingPref-style sentence, it is
   NOT fused — rewrite.
2. **No new facts.** Don't introduce objects, persons, time, or
   physical context that aren't already in the original task or
   pref_canonical.
3. **No reasoning chain.** A reader who never saw the canonical pref
   should still recover it from the words alone.
4. **One-to-one with the canonical pref.** The fused phrasing must not
   admit a competing pref interpretation (e.g. "be careful" is bad —
   maps to slow / low / avoid simultaneously).
5. **Same imperative target.** Still tells the robot what to do, just
   in different words and structure.

### Quality checklist (per S1 string)

- [ ] Could the prompt be split at a period and reduce to "task. pref.";
      if yes, fail (still in append form).
- [ ] Does *only* the target canonical pref fit?
- [ ] Removed canonical-pref keywords or substituted with manner words
      (e.g. "slow" → "carefully and gently").
- [ ] Single coherent sentence (or 2 short tightly-coupled sentences),
      length ≤ ~25 words.
- [ ] No proper nouns, no numbers unless the pref itself is numeric.

---

## Rule 2 — Reasoning paraphrasing (S2, commonsense inference)

The prompt **states the task plainly, then states a world fact /
situational context**. The pref is *not* spelled out and *not*
idiomatically equivalent — the model must run a commonsense inference
step to recover the canonical pref.

Example shape: `"<task statement>. <one-sentence context fact>."`

### What qualifies

| Cue type            | Example pref           | Example S2                                                            |
|---------------------|------------------------|-----------------------------------------------------------------------|
| Time / social       | "Speed up"             | "Guests are pulling into the driveway right now."                     |
| Object property     | "Pick by handle"       | "The mug just came out of the microwave — too hot to touch the body." |
| Workspace state     | "Move higher"          | "The countertop is wet with spills and littered with crumbs."         |
| Risk / fragility    | "Avoid the wine bottle"| "The wine bottle is half empty and tips over with the slightest nudge." |
| Geometry            | "Lay object down"      | "There is a low ceiling above the cabinet."                           |
| Personal taste      | "Prefer dairy"         | "I want something with calcium for my breakfast."                     |

### Hard constraints

1. **Inference required.** The cue must NOT contain a synonym of the
   canonical pref. If a synonym exists, it has degraded to S1.
2. **Unique recovery.** Sanity check: list the 5 nearest sibling prefs
   from the same family (e.g. all motion prefs, all colour prefs); the
   cue must uniquely select the target.
3. **Plausible scenario.** A believable real-world situation; no
   contrived "because the rule says so" framing.
4. **One commonsense hop only.** "X is fragile → handle gently → slow"
   is two hops; pick a more direct fact.
5. **No explicit pref leakage.** Never say "so go slow", "so stay
   low", etc. State the cue, nothing more.

### Quality checklist (per S2 string)

- [ ] Could a sibling pref (same family) also fit this cue? If yes, fail.
- [ ] Removed every word that appears in `pref_canonical`.
- [ ] No instructional pointer at the pref ("be careful", "handle
      gently" → borderline; "stay low", "go slow" → fail).
- [ ] Reads like something a human would actually say.

---

## Rule 1 vs Rule 2 — quick discriminator

```
canonical: "Pick by the handle."
  S1 (Rule 1, fused):    "Grip the white mug by its handle and move it to the left plate."
  S2 (Rule 2, reasoning): "Put the white mug on the left plate. The mug just came out of the microwave."
```

- If the prompt embeds the pref into the action description → Rule 1.
- If the prompt states the task plainly and adds a world fact → Rule 2.

---

## Recommended generation workflow

1. Pick a mission, read `task` (use generic / language_instructions
   wording for series 3; libero_instruction works for series 1/2) and
   `pref_canonical` from `mission_info.yaml`.
2. **Rule 1 pass:** rewrite the *whole* instruction so the pref is
   woven into the task verb/adverb/adjective. Run the Rule-1 checklist;
   reject any "task. pref." append form.
3. **Rule 2 pass:** brainstorm 3–5 plausible world facts that uniquely
   imply the same pref. Run the Rule-2 checklist; reject any that leak
   the canonical or admit sibling prefs. Pick the most natural one.
4. Cross-check: read S1 and S2 side-by-side; they should communicate
   the same pref via different mechanisms (lexical fusion vs
   inference), not just rephrase each other.
5. Commit to `dataset/series_7/language.yaml`.

---

## Recommended pref-family banks (seed pools, extend per mission)

### Motion family (Series 1)

- speed_up — Rule 1: "quickly", "in one swift motion"; Rule 2:
  guests / arrivals / deadlines.
- slow_down — Rule 1: "carefully", "ease X into Y"; Rule 2: fragility,
  softening contents, sleeping baby.
- low_path — Rule 1: "keep the arm low", "slide along the surface";
  Rule 2: low cabinet, hanging lamp, overhead camera.
- high_path — Rule 1: "lift well clear of the surface"; Rule 2: wet
  counter, crumbs, broken glass.
- straight_path — Rule 1: "along the most direct line"; Rule 2:
  cramped space, wires.
- avoid_object — Rule 1: "routing the arm clear of X"; Rule 2: X is
  unstable / hot / fragile.
- max_acc / smoothness — Rule 1: "smoothly", "no jolts"; Rule 2:
  stack of plates, full glass nearby.
- rotation — Rule 1: "turn it as you carry"; Rule 2: demo / trial of
  wrist motion.

### Skill family (Series 2)

- pick_<edge>/<side>/<front>/<back> — Rule 1: "grip from <dir>",
  "approach from <dir>"; Rule 2: opposite side blocked / chipped /
  sticky / hot / pressed against another item.
- pick_handle — Rule 1: "by its handle"; Rule 2: hot body, wet body,
  greasy body.
- pick_top_angle — Rule 1: "from above with the gripper tilted";
  Rule 2: wedged between bottles, only opening is from above.
- throw — Rule 1: "toss"; Rule 2: obstacle in between, can't reach
  across.
- rotate_when_place / lay_down — Rule 1: "twist as you set",
  "on its side"; Rule 2: compartment narrower than object,
  low ceiling overhead.

### Profile family (Series 3)

- prefer_<colour/size/shape> — Rule 1: name the chosen object's
  attribute directly ("the green X", "the smaller bowl"); Rule 2:
  personal taste, decor consistency, scenario need.
- prefer_<category> (dairy / canned / boxed / sour / sweet / bottled)
  — Rule 1: name the specific item; Rule 2: nutrition reason, recipe
  reason, packaging reason.
- drawer_for_<X> (food / kitchenware) — Rule 1: name the drawer
  directly ("top drawer", "middle drawer"); Rule 2: state the
  household rule for that drawer.
- double_portion — Rule 1: "serve a double portion: ..."; Rule 2:
  guest asked for seconds, larger group.

---

## Things NOT to do

- Don't translate into another language (not under our axis).
- Don't add a second pref ("be quick AND stay low") — keeps the axis
  single-pref.
- Don't reference series, mission ids, or rule numbers inside the
  language string.
- Don't make Rule 2 cues so vague that the pref is unrecoverable
  (e.g. "It's been a long day"). Rule 2 must still be uniquely
  solvable.
- Don't author Rule 1 by appending the canonical pref with a
  comma/period — that is just SingPref formatting.
