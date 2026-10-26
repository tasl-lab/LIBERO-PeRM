"""One-shot script: rename preferences in all mission_info.yaml.

Strategy (v2):
- For each perm_X_Y/libero_ZZZ/mission_info.yaml resolved to mission_id "X-Y_ZZZ",
  unconditionally replace non-empty `*_pref_lan:` and `description:` (incl. list-item
  `- description:`) with the canonical new preference text.
- Preserves trailing yaml comments (`# ...`).
- Skips LIBERO_mission/* and perm_m1_4/libero_053 (target_obj=frypan, manual).
- Handles inline trailing comments like `key: value #comment`.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PREFIX_MAP = {
    "1-1": "Speed up",
    "1-2": "Slow down",
    "1-3": "Avoid sudden acceleration",
    "1-5": "Close to ground when moving",
    "1-6": "Move straight",
    "1-7": "Move higher",
    "1-8": "Rotate when moving",
    "2-1": "Pick from right edge",
    "2-2": "Pick from back edge",
    "2-3": "Pick by handle",
    "2-4": "Pick from top at an angle",
    "2-5": "Pick from left side",
    "2-6": "Pick from right side",
    "2-7": "Pick from horizontal side",
    "2-8": "Push instead of pick-and-put",
    "2-9": "Drag instead of pick-and-put",
    "2-10": "Throw instead of placing",
    "2-11": "Pick from front",
    "2-12": "Rotate when placing",
    "2-13": "Pick from left edge",
    "2-14": "Lay object down when placing",
    "3-1": "Prefer green objects",
    "3-2": "Prefer red objects",
    "3-3": "Prefer yellow objects",
    "3-4": "Prefer blue objects",
    "3-5": "Prefer bigger objects",
    "3-6": "Upper drawer for food and drink",
    "3-7": "Middle drawer for kitchenware",
    "3-8": "Serve double portion",
    "3-9": "Prefer dairy products",
    "3-10": "Prefer sweet drinks",
    "3-11": "Prefer sour sauces",
    "3-12": "Prefer boxed objects",
    "3-13": "Prefer bottled objects",
    "3-14": "Prefer canned food",
    "3-15": "Prefer smaller objects",
}

SPECIFIC = {
    "1-4_034": "Keep away from wine bottle",
    "1-4_053": "Keep away from stove",
}

SKIP_FILES = {
    "series_4/perm_4_1/libero_053/mission_info.yaml",  # target_obj=frypan; manual review (was perm_m1_4)
}


def new_pref_for(series_task, libero_id):
    full = f"{series_task}_{libero_id}"
    if full in SPECIFIC:
        return SPECIFIC[full]
    return PREFIX_MAP.get(series_task)


def parse_dir(suite_dir, libero_dir):
    if suite_dir == "LIBERO_mission":
        return None
    m = re.match(r"perm_(m?)(\d+)_(\d+)$", suite_dir)
    if not m:
        return None
    series_task = f"{m.group(2)}-{m.group(3)}"
    libero_id = libero_dir.replace("libero_", "")
    return series_task, libero_id


PREF_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dash>-\s+)?"
    r"(?P<key>low_pref_lan|high_pref_lan|description):"
    r"\s*(?P<value>.*?)\s*$"
)


def split_value_comment(rest):
    """Split 'value #comment' → (value, ' #comment'). Handle quoted strings carefully."""
    # find ' #' that is NOT inside quotes
    in_q = None
    for i, ch in enumerate(rest):
        if in_q:
            if ch == in_q:
                in_q = None
            continue
        if ch in ("'", '"'):
            in_q = ch
            continue
        if ch == "#" and i > 0 and rest[i - 1] in (" ", "\t"):
            # comment starts at i, preserve from preceding space
            # find start of whitespace run before #
            j = i - 1
            while j > 0 and rest[j - 1] in (" ", "\t"):
                j -= 1
            return rest[:j].rstrip(), rest[j:]
    return rest.rstrip(), ""


def strip_quotes(v):
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def replace_pref_in_file(path, new_pref):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        raw = line.rstrip("\n")
        m = PREF_LINE_RE.match(raw)
        if not m:
            continue
        indent = m.group("indent")
        dash = m.group("dash") or ""
        key = m.group("key")
        value_full = m.group("value")
        if value_full == "":
            continue
        value, comment = split_value_comment(value_full)
        clean = strip_quotes(value)
        if clean in ("", "''", '""'):
            continue
        if clean == new_pref:
            continue
        new_line = f"{indent}{dash}{key}: {new_pref}{comment}"
        if line.endswith("\n"):
            new_line += "\n"
        lines[i] = new_line
        changed = True
    if changed:
        path.write_text("".join(lines))
    return changed


def main():
    edited, skipped = [], []
    for path in sorted(ROOT.rglob("mission_info.yaml")):
        rel = str(path.relative_to(ROOT))
        if rel in SKIP_FILES:
            skipped.append(f"{rel}  (manual skip)")
            continue
        parts = path.relative_to(ROOT).parts
        # Post-migration: parts = (series_X, perm_X_Y, libero_*, mission_info.yaml)
        if len(parts) < 4 or not parts[0].startswith("series_"):
            continue
        suite_dir, libero_dir = parts[1], parts[2]
        parsed = parse_dir(suite_dir, libero_dir)
        if parsed is None:
            skipped.append(f"{rel}  (LIBERO_mission)")
            continue
        series_task, libero_id = parsed
        new_pref = new_pref_for(series_task, libero_id)
        if new_pref is None:
            skipped.append(f"{rel}  (no mapping for {series_task})")
            continue
        if replace_pref_in_file(path, new_pref):
            edited.append(f"{rel}  -> {new_pref!r}")
    print(f"=== Edited {len(edited)} files ===")
    for e in edited:
        print(e)
    print(f"\n=== Skipped {len(skipped)} ===")


if __name__ == "__main__":
    main()
