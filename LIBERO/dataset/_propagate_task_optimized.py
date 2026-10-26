"""One-shot script: insert/update `optimized_instruction:` field in mission_info.yaml.

Strategy:
- Mapping: mission_id ("X-Y_ZZZ") → optimized task text
- For each perm_X_Y/libero_*/mission_info.yaml (skip perm_m*, LIBERO_mission/*):
  - Find `language_instructions:` line (and YAML continuation lines indented with spaces)
  - If `optimized_instruction:` line already exists → replace value
  - Else insert `optimized_instruction: <text>` immediately after the
    language_instructions block

Style follows VLA_data/SinglePrefDataTest/tasksuite.yaml: drop "the" / "the cabinet"
trailing clauses, unify "Pick X and place/put it in Y" → "Put X in Y", drop "both",
shorten "compartment of the caddy" → "caddy compartment".
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# mission_id → optimized task text
OPTIMIZED = {
    # Series 1
    "1-1_004": "Put alphabet soup and cream cheese in basket",
    "1-1_006": "Put cream cheese and butter in basket",
    "1-2_062": "Put milk in basket",
    "1-2_065": "Put alphabet soup in tray",
    "1-2_113": "Put alphabet soup in tray",  # perm dir exists; same task
    "1-3_107": "Put cream cheese on bowl",
    "1-3_109": "Put wine bottle on top of drawer",
    "1-4_034": "Put black bowl in bottom drawer",
    "1-4_053": "Put white bowl on top of cabinet",
    "1-5_034": "Put black bowl in bottom drawer",
    "1-5_070": "Put left black bowl in tray",
    "1-5_109": "Put wine bottle on top of drawer",
    "1-5_114": "Put wine bottle on top of cabinet",
    "1-6_067": "Put cream cheese in tray",
    "1-6_071": "Put chocolate pudding in tray",
    "1-6_105": "Put chocolate pudding in tray",
    "1-7_015": "Put chocolate pudding in top drawer and close it",
    "1-7_039": "Put black bowl in top drawer",
    "1-7_104": "Put black bowl in top drawer",
    "1-8_049": "Turn off stove",
    "1-8_061": "Put butter in basket",
    "1-8_114": "Put wine bottle on top of cabinet",
    # Series 2
    "2-1_019": "Put black bowl on plate",
    "2-1_077": "Put white mug on left plate",
    "2-2_019": "Put black bowl on plate",
    "2-2_077": "Put white mug on left plate",
    "2-3_075": "Put red mug on left plate",
    "2-3_077": "Put white mug on left plate",
    "2-4_112": "Put BBQ sauce in basket",
    "2-4_119": "Put salad dressing in basket",
    "2-5_112": "Put BBQ sauce in basket",
    "2-5_116": "Put ketchup in basket",
    "2-6_042": "Put ketchup in top drawer",
    "2-6_119": "Put salad dressing in basket",
    "2-7_ood1": "Put blue plate on top of microwave",
    "2-7_ood2": "Put blue plate on top of microwave",
    "2-8_ood3": "Move basket to left",
    "2-8_ood4": "Move basket to left",
    "2-9_ood3": "Move basket to left",
    "2-9_ood4": "Move basket to left",
    "2-10_015": "Put chocolate pudding in top drawer and close it",
    "2-10_065": "Put alphabet soup in tray",
    "2-11_058": "Put ketchup in basket",
    "2-11_062": "Put milk in basket",
    "2-11_116": "Put ketchup in basket",
    "2-11_118": "Put milk in basket",
    "2-12_009": "Put book in right caddy compartment",
    "2-12_085": "Put book in right caddy compartment",
    "2-12_093": "Put book in right caddy compartment",
    "2-13_001": "Put black bowl in bottom drawer and close it",
    "2-13_034": "Put black bowl in bottom drawer",
    "2-14_068": "Put ketchup in tray",
    "2-14_109": "Put wine bottle on top of cabinet",
    # Series 3
    "3-1_004_1": "Put alphabet soup and cream cheese in basket",
    "3-1_006_1": "Put cream cheese and butter in basket",
    "3-2_001_1": "Put bowl in bottom drawer and close it",
    "3-2_097_1": "Put yellow and white mug to right of caddy",
    "3-3_019_1": "Put black bowl on a plate",
    "3-3_084_1": "Put book in back caddy compartment",
    "3-3_117_2": "Put book in back caddy compartment",
    "3-4_058_1": "Put ketchup in basket",
    "3-4_081_1": "Put red mug on a plate",
    "3-4_104_1": "Put ketchup in basket",
    "3-5_023_1": "Put front black bowl on a plate",
    "3-5_064_1": "Put a tomato sauce in basket",
    "3-5_120_1": "Put tomato sauce in basket",
    "3-6_012_1": "Put one item in top drawer",
    "3-6_036_1": "Put wine bottle in bottom drawer",
    "3-6_101_1": "Put an item inside",
    "3-7_018_1": "Open drawer and put bowl in it",
    "3-7_101_1": "Open one drawer layer and put bowl inside",
    "3-8_007_1": "Put white mug on left plate",
    "3-8_070_1": "Put left black bowl in tray",
    "3-8_081_2": "Put red mug on a plate",
    "3-9_063_1": "Put a drink in basket",
    "3-9_118_1": "Put a drink in basket",
    "3-10_062_1": "Put a drink in basket",
    "3-10_117_1": "Put a drink in basket",
    "3-11_072_1": "Put a sauce in tray",
    "3-11_112_1": "Put a sauce in basket",
    "3-12_036_2": "Put a drink in bottom drawer",
    "3-12_060_1": "Put a food in basket",
    "3-12_109_1": "Put a food or drink on top of drawer",
    "3-13_046_1": "Put a liquid container on plate",
    "3-13_120_1": "Put a tomato-flavored sauce in basket",
    "3-14_107_1": "Put a food in bowl",
    "3-14_114_1": "Put a food in basket",
    "3-15_040_1": "Put a bowl on plate",
    "3-15_041_1": "Put a bowl on top of cabinet",
    "3-15_106_1": "Put a bowl on top of cabinet",
}


def parse_dir(suite_dir, libero_dir):
    """Return mission_id 'X-Y_ZZZ' or None to skip."""
    if suite_dir == "LIBERO_mission":
        return None
    # skip multi-pref
    if suite_dir.startswith("perm_m"):
        return None
    m = re.match(r"perm_(\d+)_(\d+)$", suite_dir)
    if not m:
        return None
    series_task = f"{m.group(1)}-{m.group(2)}"
    libero_id = libero_dir.replace("libero_", "")
    return f"{series_task}_{libero_id}"


def find_lang_block_end(lines, start_idx):
    """Given `language_instructions:` line index, return index of last line in block.
    Block = the line itself + any indented continuation (yaml folded multi-line).
    """
    end = start_idx
    for j in range(start_idx + 1, len(lines)):
        # continuation: indented with whitespace, no new key colon at top-level
        line = lines[j]
        if not line.strip():
            break
        if line[0] in (" ", "\t") and not re.match(r"^\s*\w+\s*:", line):
            # continuation
            end = j
        elif line.startswith("  ") and re.match(r"^\s+\w+\s*:", line):
            # nested key (e.g., subtask_offset:); not continuation
            break
        else:
            break
    return end


def process_file(path, optimized_text):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    # Find existing optimized_instruction line
    opt_idx = None
    lang_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*language_instructions\s*:", line) and lang_idx is None:
            lang_idx = i
        if re.match(r"^\s*optimized_instruction\s*:", line) and opt_idx is None:
            opt_idx = i
    if lang_idx is None:
        return False, "no language_instructions"
    if opt_idx is not None:
        # Replace value
        ending = "\n" if lines[opt_idx].endswith("\n") else ""
        new_line = f"optimized_instruction: {optimized_text}{ending}"
        if lines[opt_idx] == new_line:
            return False, "already up to date"
        lines[opt_idx] = new_line
        path.write_text("".join(lines))
        return True, "updated"
    # Insert after language_instructions block
    end = find_lang_block_end(lines, lang_idx)
    # ensure last line has a trailing newline
    if not lines[end].endswith("\n"):
        lines[end] = lines[end] + "\n"
    insert_line = f"optimized_instruction: {optimized_text}\n"
    lines.insert(end + 1, insert_line)
    path.write_text("".join(lines))
    return True, "inserted"


def main():
    edited = []
    skipped = []
    missing = []
    for path in sorted(ROOT.rglob("mission_info.yaml")):
        rel = path.relative_to(ROOT)
        # Post-migration: parts = (series_X, perm_X_Y, libero_*, mission_info.yaml)
        if len(rel.parts) < 4 or not rel.parts[0].startswith("series_"):
            continue
        suite_dir, libero_dir = rel.parts[1], rel.parts[2]
        mid = parse_dir(suite_dir, libero_dir)
        if mid is None:
            skipped.append(f"{rel}  (LIBERO_mission or m-series)")
            continue
        if mid not in OPTIMIZED:
            missing.append(f"{rel}  (no mapping for {mid})")
            continue
        ok, msg = process_file(path, OPTIMIZED[mid])
        if ok:
            edited.append(f"{rel}  -> {OPTIMIZED[mid]!r}  [{msg}]")
        else:
            skipped.append(f"{rel}  ({msg})")
    print(f"=== Edited {len(edited)} ===")
    for e in edited:
        print(e)
    print(f"\n=== Missing mapping {len(missing)} ===")
    for m in missing:
        print(m)
    print(f"\n=== Skipped {len(skipped)} ===")


if __name__ == "__main__":
    main()
