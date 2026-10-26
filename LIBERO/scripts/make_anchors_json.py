#!/usr/bin/env python
"""Build {mission: {pref_type: {pref_anchor, naive_anchor}}} from a
score_motion_demos raw cache (median pref raw / median naive raw)."""
import json, sys, os, collections
import numpy as np

cache_path = sys.argv[1] if len(sys.argv) > 1 else "logs/motion_raw_cache_final.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "logs/anchors.json"
cache = json.load(open(cache_path))
pref, naive = collections.defaultdict(dict), collections.defaultdict(dict)
for key, v in cache.items():
    path = key.split("|")[0]
    mid = os.path.basename(path)[8:-5]
    tgt = naive if "_naive" in path else pref
    for ptype, raws in v["raws"].items():
        vals = [r for r in raws if r is not None]
        if vals:
            tgt[mid][ptype] = float(np.median(vals))
out = {}
for mid in pref:
    for ptype, pv in pref[mid].items():
        d = {"pref_anchor": pv}
        if ptype in naive.get(mid, {}):
            d["naive_anchor"] = naive[mid][ptype]
        out.setdefault(mid, {})[ptype] = d
json.dump(out, open(out_path, "w"), indent=1)
print(f"wrote {out_path}: {len(out)} missions")
for mid, d in out.items():
    for pt, a in d.items():
        print(f"  {mid:10} {pt:16} pref={a['pref_anchor']:.5g} naive={a.get('naive_anchor', float('nan')):.5g}")
