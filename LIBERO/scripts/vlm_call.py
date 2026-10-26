#!/usr/bin/env python
"""Record the VLM judge's per-tile calls for a contact sheet.
Usage: python scripts/vlm_call.py 2-1_019 1111111111 0000000000   (pref calls, naive calls; 1 = preference visibly satisfied)
"""
import json, os, os, sys
HR = os.environ.get("PERM_SHEETS", "/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2/sheets")
mid, pref = sys.argv[1], [int(c) for c in sys.argv[2]]
naive = [int(c) for c in sys.argv[3]] if len(sys.argv) > 3 else []
json.dump({"mission": mid, "pref": pref, "naive": naive, "judge": "claude (frames at the skill node)"}, open(os.path.join(HR, f"{mid}.vlm.json"), "w"))
m = json.load(open(os.path.join(HR, f"{mid}.json")))
agree = tot = 0
for side, calls in (("pref", pref), ("naive", naive)):
    for x, c in zip(m.get(side, []), calls):
        if x.get("verdict") is None: continue
        tot += 1; agree += int((x["verdict"] >= 100) == bool(c))
print(f"{mid}: judge vs metric agree {agree}/{tot}")
