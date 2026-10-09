#!/usr/bin/env python3
"""Collect the Figure 4 numbers (SR / SA per scale cell and generalization branch) from evalresult.xlsx.

Usage: python fig4_collect.py [--root Policy_output/lora_ft] [--out fig4.csv]
Branches: in-domain = liberoscale/H_<cell>_*_4trial, unseen-layout = liberoscale_layout_ood/H_<cell>_*,
unseen-layout+different-task = mean over liberoscale_pref_ood/H_<cell>_*_pref_ood_{speed,left,red,multi}.
"""
import argparse, glob, os, pandas as pd

CELLS = [("M1", 250), ("M2", 750), ("M3", 2250), ("M4", 5250), ("L1", 2250), ("L3", 2250), ("L4", 2295)]

def rollup(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    df = pd.read_excel(hits[-1], sheet_name="rollup")          # latest evalresult*.xlsx
    row = df[df.series.astype(str) == "ALL"].iloc[0]
    return 100 * float(row.sr), float(row.final_sa)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="Policy_output/lora_ft"); ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for cell, n in CELLS:
        rec = {"cell": cell, "demos": n}
        v = rollup(f"{a.root}/liberoscale/H_{cell}_*_4trial/*/evalresult*.xlsx")
        rec.update(indomain_sr=v[0] if v else None, indomain_sa=v[1] if v else None)
        v = rollup(f"{a.root}/liberoscale_layout_ood/H_{cell}_*_layout_ood/*/evalresult*.xlsx")
        rec.update(layout_ood_sr=v[0] if v else None, layout_ood_sa=v[1] if v else None)
        prefs = [rollup(f"{a.root}/liberoscale_pref_ood/H_{cell}_*_pref_ood_{p}/*/evalresult*.xlsx")
                 for p in ("speed", "left", "red", "multi")]
        prefs = [p for p in prefs if p]
        rec.update(pref_ood_sr=sum(p[0] for p in prefs) / len(prefs) if prefs else None,
                   pref_ood_sa=sum(p[1] for p in prefs) / len(prefs) if prefs else None, pref_ood_n=len(prefs))
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
