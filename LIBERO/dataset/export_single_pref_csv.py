"""Export single preference table as CSV from perm_<p>_<s>/libero_<N>/mission_info.yaml."""
import csv
import re
from pathlib import Path

import yaml

DATASET_DIR = Path(__file__).parent
OUT_CSV = DATASET_DIR / "single_preference.csv"

CATEGORY_MAP = {1: "Motion preference", 2: "Skill preference", 3: "Object selection preference"}


def natural_layout_key(name: str):
    m = re.match(r"libero_(.+)", name)
    if not m:
        return (9, name)
    tail = m.group(1)
    ood = re.match(r"ood(\d+)", tail)
    if ood:
        return (1, int(ood.group(1)))
    base = re.match(r"(\d+)(?:_(\d+))?", tail)
    if base:
        return (0, int(base.group(1)), int(base.group(2)) if base.group(2) else 0)
    return (2, tail)


def collect():
    perm_dirs = []
    # Post-migration layout: dataset/series_X/perm_X_Y/
    for series_p in DATASET_DIR.iterdir():
        if not (series_p.is_dir() and series_p.name.startswith("series_")):
            continue
        for p in series_p.iterdir():
            if not (p.is_dir() and p.name.startswith("perm_")):
                continue
            m = re.match(r"perm_(\d+)_(\d+)$", p.name)
            if not m:
                continue  # skip perm_m*_* (multi-pref) and other variants
            perm_dirs.append((int(m.group(1)), int(m.group(2)), p))
    perm_dirs.sort()
    perm_dirs = [p for _, _, p in perm_dirs]

    rows_by_series: dict[int, list[list[str]]] = {1: [], 2: [], 3: []}

    for perm in perm_dirs:
        m = re.match(r"perm_(\d+)_(\d+)", perm.name)
        if not m:
            continue
        series, sub = int(m.group(1)), int(m.group(2))
        pref_no = f"{series}-{sub}"

        libero_dirs = sorted(
            (d for d in perm.iterdir() if d.is_dir() and d.name.startswith("libero_")),
            key=lambda d: natural_layout_key(d.name),
        )
        for lib in libero_dirs:
            layout_no = lib.name.replace("libero_", "")
            if layout_no.endswith("_deprecated"):
                continue
            yaml_path = lib / "mission_info.yaml"
            if not yaml_path.exists():
                continue
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}

            language = (data.get("language_instructions") or "").strip().replace("\n", " ")

            def first_desc(key):
                items = data.get(key) or []
                if items and isinstance(items[0], dict):
                    return (items[0].get("description") or "").strip()
                return ""

            if series == 1:
                pref_text = first_desc("low_prefs") or (data.get("low_pref_lan") or "").strip()
            elif series == 2:
                pref_text = first_desc("high_prefs") or (data.get("high_pref_lan") or "").strip()
            else:  # series 3
                pref_text = (data.get("high_pref_lan") or "").strip()
            preference = f"Preference: {pref_text}" if pref_text else ""

            rows_by_series[series].append([pref_no, layout_no, language, preference, ""])

    return rows_by_series


def main():
    rows_by_series = collect()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Pref No", "Layout No", "Language task", "Preference", "TODO"])
        for s in (1, 2, 3):
            if not rows_by_series[s]:
                continue
            w.writerow([CATEGORY_MAP[s], "", "", "", ""])
            for row in rows_by_series[s]:
                w.writerow(row)
    total = sum(len(v) for v in rows_by_series.values())
    print(f"Wrote {total} mission rows -> {OUT_CSV}")
    for s in (1, 2, 3):
        print(f"  Series {s} ({CATEGORY_MAP[s]}): {len(rows_by_series[s])}")


if __name__ == "__main__":
    main()
