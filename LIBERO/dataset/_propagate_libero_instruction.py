"""One-shot script: insert/update `libero_instruction:` field in mission_info.yaml.

Strategy:
- For each perm_X_Y/libero_*/mission_info.yaml (skip perm_m*, LIBERO_mission/*):
  - Find the unique top-level `*.bddl` in the mission dir (Series 3 also has
    no_pref_layout/*.bddl with the same filename — we use the top-level one).
  - Derive instruction text from bddl stem:
      strip leading `^[A-Z][A-Z_]*SCENE\d+_` (Series 1/2 scene prefix), then
      `_` → space.
    e.g. `LIVING_ROOM_SCENE1_put_both_..._in_the_basket.bddl`
       → "put both the alphabet soup and the cream cheese box in the basket"
  - Insert/update top-level `libero_instruction:` field:
      * If line exists → replace value.
      * Else insert immediately after `optimized_instruction:` (if present),
        else after `language_instructions:` block.

Usage:
    python dataset/_propagate_libero_instruction.py             # dry-run
    python dataset/_propagate_libero_instruction.py --apply     # write
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCENE_PREFIX_RE = re.compile(r"^[A-Z][A-Z_]*SCENE\d+_")


def parse_dir(suite_dir: str) -> bool:
    """Return True if this perm dir should be processed."""
    if suite_dir == "LIBERO_mission":
        return False
    if suite_dir.startswith("perm_m"):
        return False
    return bool(re.match(r"perm_(\d+)_(\d+)$", suite_dir))


def derive_instruction(bddl_stem: str) -> str:
    stripped = SCENE_PREFIX_RE.sub("", bddl_stem)
    return stripped.replace("_", " ")


def find_top_bddl(mission_dir: Path) -> Path | None:
    bddls = sorted(p for p in mission_dir.iterdir() if p.suffix == ".bddl" and p.is_file())
    if not bddls:
        return None
    if len(bddls) > 1:
        # Should not happen — keep first deterministically and let caller see.
        return bddls[0]
    return bddls[0]


def process_file(path: Path, instruction_text: str) -> tuple[bool, str]:
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    libero_idx = None
    opt_idx = None
    lang_idx = None
    for i, line in enumerate(lines):
        if lang_idx is None and re.match(r"^\s*language_instructions\s*:", line):
            lang_idx = i
        if opt_idx is None and re.match(r"^\s*optimized_instruction\s*:", line):
            opt_idx = i
        if libero_idx is None and re.match(r"^\s*libero_instruction\s*:", line):
            libero_idx = i

    new_line = f"libero_instruction: {instruction_text}\n"

    if libero_idx is not None:
        if lines[libero_idx] == new_line:
            return False, "already up to date"
        lines[libero_idx] = new_line
        path.write_text("".join(lines))
        return True, "updated"

    # Insert: prefer after optimized_instruction, else after language_instructions block.
    anchor_idx = opt_idx if opt_idx is not None else lang_idx
    if anchor_idx is None:
        return False, "no language_instructions / optimized_instruction anchor"

    # Walk to end of block (handle YAML folded continuation lines).
    end = anchor_idx
    for j in range(anchor_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            break
        if line[0] in (" ", "\t") and not re.match(r"^\s*\w+\s*:", line):
            end = j
            continue
        break

    if not lines[end].endswith("\n"):
        lines[end] = lines[end] + "\n"
    lines.insert(end + 1, new_line)
    path.write_text("".join(lines))
    return True, "inserted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    args = ap.parse_args()

    edited: list[str] = []
    skipped: list[str] = []
    no_bddl: list[str] = []

    for path in sorted(ROOT.rglob("mission_info.yaml")):
        rel = path.relative_to(ROOT)
        # Post-migration: dataset/series_X/perm_X_Y/libero_*/mission_info.yaml
        if len(rel.parts) < 3 or not rel.parts[0].startswith("series_"):
            continue
        suite_dir = rel.parts[1]
        if not parse_dir(suite_dir):
            continue

        mission_dir = path.parent
        bddl = find_top_bddl(mission_dir)
        if bddl is None:
            no_bddl.append(str(rel))
            continue
        instruction = derive_instruction(bddl.stem)

        # In dry-run, simulate process_file without writing.
        if args.apply:
            changed, msg = process_file(path, instruction)
        else:
            text = path.read_text()
            lines = text.splitlines(keepends=True)
            existing = next((i for i, l in enumerate(lines) if re.match(r"^\s*libero_instruction\s*:", l)), None)
            if existing is None:
                changed, msg = True, "would insert"
            else:
                expected = f"libero_instruction: {instruction}\n"
                if lines[existing] == expected:
                    changed, msg = False, "already up to date"
                else:
                    changed, msg = True, "would update"

        if changed:
            edited.append(f"{rel}\n    bddl: {bddl.name}\n    -> {instruction!r}  [{msg}]")
        else:
            skipped.append(f"{rel}  ({msg})")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== [{mode}] Edited / would-edit: {len(edited)} ===")
    for e in edited:
        print(e)
    print(f"\n=== Skipped (no change): {len(skipped)} ===")
    for s in skipped[:20]:
        print(s)
    if len(skipped) > 20:
        print(f"... ({len(skipped) - 20} more)")
    if no_bddl:
        print(f"\n=== Missions with no bddl ({len(no_bddl)}) ===")
        for n in no_bddl:
            print(n)


if __name__ == "__main__":
    main()
