#!/usr/bin/env python3
"""Split the single PrefCompose eval into the four Table 4 cells.

Layouts 4-3 / 4-4 were trained on {nopref, A, B}  → eval AB   = composition
Layouts 4-5 / 4-6 were trained on {nopref, A, AB} → eval B    = decomposition
"In-dom." = the three trained conditions of the same layouts, nopref included (a successful nopref episode
counts as SA = 1: there is no preference to violate) — this reproduces the paper's 14/24 and 16/24.

Usage: python table4_cells.py [path/to/evalresult.xlsx]
"""
import sys, pandas as pd

DEFAULT = ("Policy_output/lora_ft/pref-compose/PrefCompose_step21k_4trial/"
           "pi05_prefcompose_run1_step21000/evalresult.xlsx")
SPLIT = {  # name: (in-domain missions, eval missions)
    "composition":   (["4-3_116_combo_nopref", "4-3_116_combo_A", "4-3_116_combo_B",
                       "4-4_001_combo_nopref", "4-4_001_combo_A", "4-4_001_combo_B"],
                      ["4-3_116_combo_AB", "4-4_001_combo_AB"]),
    "decomposition": (["4-5_118m_combo_nopref", "4-5_118m_combo_A", "4-5_118m_combo_AB",
                       "4-6_068_combo_nopref", "4-6_068_combo_A", "4-6_068_combo_AB"],
                      ["4-5_118m_combo_B", "4-6_068_combo_B"]),
}

def cell(df, missions):
    d = df[df.mission_id.isin(missions)]
    n_succ, n_total = int(d.n_succ.sum()), int(d.n_total.sum())
    sr = n_succ / n_total if n_total else float("nan")
    # SA = SR x raw SA, success-gated (Sec. 3.4); missions with no success contribute 0
    def sa_factor(r):
        if r.mission_id.endswith("_nopref"):
            return 1.0
        return r.mapped_sa / 100 if pd.notna(r.mapped_sa) else 0.0
    sa = sum((r.n_succ / r.n_total) * sa_factor(r) for r in d.itertuples()) / len(d) if len(d) else float("nan")
    return 100 * sr, 100 * sa, f"{n_succ}/{n_total}"

pm = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else DEFAULT, sheet_name="per_mission")
print("| setting | In-dom. SR | In-dom. SA | Eval SR | Eval SA |")
print("|---|---|---|---|---|")
for name, (indom, ev) in SPLIT.items():
    a, b = cell(pm, indom), cell(pm, ev)
    print(f"| {name} | {a[0]:.1f} ({a[2]}) | {a[1]:.1f} | {b[0]:.1f} ({b[2]}) | {b[1]:.1f} |")
