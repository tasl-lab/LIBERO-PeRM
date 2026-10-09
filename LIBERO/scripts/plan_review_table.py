#!/usr/bin/env python
"""Series-3 (plan_choice) human-review summary: two missions per family + a table of everything.

Reads /tmp/s3gen/finish_<id>.json, table_<id>.csv, pref_<id>.log / naive_<id>.log and writes
  <HR>/REVIEW_NOTES.md              all-missions table + the per-family picks
  <HR>/NN_<family>__<mission>.mp4   2 videos per family (copied from <HR>/all/)
Usage: cd LIBERO && python scripts/plan_review_table.py [--hr DIR]
"""
import argparse, csv, glob, json, os, re, shutil, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
HR_DEFAULT = "/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S3"
LOGD = "/tmp/s3gen"
FAMILY_NAME = {1: "green", 2: "red", 3: "yellow", 4: "blue", 5: "bigger", 6: "food_upper_drawer", 7: "kitchenware_middle_drawer",
               8: "double_portion", 9: "dairy", 10: "sweet_drink", 11: "sour_sauce", 12: "boxed", 13: "bottled", 14: "canned", 15: "smaller"}
SKIP = re.compile(r"(_nopref$|^097_|_1c$)")   # scratch variants / one-off copies


def missions():
    out = []
    for d in sorted(glob.glob(os.path.join(LIBERO, "dataset/series_3/perm_3_*/libero_*"))):
        lid = os.path.basename(d)[len("libero_"):]
        if SKIP.search(lid):
            continue
        fam = int(re.search(r"perm_3_(\d+)", d).group(1))
        out.append((fam, f"3-{fam}_{lid}", d))
    return sorted(out, key=lambda x: (x[0], x[1]))


def sr_from_log(path):
    if not os.path.exists(path):
        return "-"
    hits = re.findall(r"success rate: (\d+/\d+)", open(path, errors="ignore").read())
    return hits[-1] if hits else "-"


def branch(lp):
    p = lp.get("params") or {}
    if p.get("mode") == "target":
        return f"{p.get('carried')} -> {'/'.join(p.get('pref_targets', []))}", f"{p.get('carried')} -> {'/'.join(p.get('naive_targets', []))}"
    return "/".join(p.get("pref_objs", [])) or "-", "/".join(p.get("naive_objs", [])) or "(没有多抬那个)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hr", default=HR_DEFAULT); a = ap.parse_args()
    rows = []
    for fam, mid, d in missions():
        y = yaml.safe_load(open(os.path.join(d, "mission_info.yaml"))) or {}
        lp = (y.get("low_prefs") or [{}])[0]
        if lp.get("type") != "plan_choice":
            continue
        pb, nb = branch(lp)
        r = {"fam": fam, "mid": mid, "pref_lan": str(y.get("high_pref_lan", "")).split("#")[0].strip(), "mode": (lp.get("params") or {}).get("mode", "-"),
             "pref_branch": pb, "naive_branch": nb, "pref_sr": sr_from_log(f"{LOGD}/pref_{mid}.log"), "naive_sr": "-",
             "ok": False, "note": "", "sat_p": None, "sat_n": None, "vlm": None, "video": "-"}
        if r["pref_sr"] == "-" and os.path.exists(f"/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_{mid}.hdf5"):
            r["pref_sr"] = "100 (earlier run)"
        jf = f"{LOGD}/finish_{mid}.json"
        if os.path.exists(jf):
            j = json.load(open(jf)); r["ok"] = bool(j.get("ok"))
            m = re.search(r"naive SR (\d+/\d+)", j.get("pair", "")); r["naive_sr"] = m.group(1) if m else "-"
            r["note"] = "" if r["ok"] else f"{j.get('stage')}: {j.get('note')}"
        else:
            r["naive_sr"] = sr_from_log(f"{LOGD}/naive_{mid}.log")
        tf = f"{LOGD}/table_{mid}.csv"
        if os.path.exists(tf):
            for c in csv.DictReader(open(tf)):
                fl = lambda v: float(v) if v not in ("", None) else None
                r["sat_p"] = fl(c.get("sat_pref")); r["sat_n"] = fl(c.get("sat_naive"))
        sj, vj = os.path.join(a.hr, "sheets", f"{mid}.json"), os.path.join(a.hr, "sheets", f"{mid}.vlm.json")
        if os.path.exists(sj) and os.path.exists(vj):
            m, v = json.load(open(sj)), json.load(open(vj)); agree = tot = 0
            for side in ("pref", "naive"):
                for x, call in zip(m.get(side, []), v.get(side, [])):
                    if x.get("verdict") is None: continue
                    tot += 1; agree += int((x["verdict"] >= 100) == bool(call))
            r["vlm"] = f"{agree}/{tot}" if tot else None
        cand = f"all/{mid}_pref_vs_nopref.mp4"
        if os.path.exists(os.path.join(a.hr, cand)):
            r["video"] = cand
        rows.append(r)
    md = ["# Profile-preference（series_3）待审：15 家 × 2 mission = 30 个视频\n",
          "每个 mission：pref demo 是 task_plan（带偏好的分支），naive 是 naive_plan 在同一个 init 上跑出来的（no_pref bddl 的目标 = naive 分支）。"
          "打分是节点式离散 0/100：pref 和 naive 两条 plan 分岔的那一步（抬起哪个物体 / 放进哪个区域）判一次，之前没分、之后保持。"
          "「判据满足」= 10 条 pref / 10 条 naive 里判 100 的比例；「VLM 一致」= 我按 sheets/<mission>.png 逐格看图判的结果与 metric 一致的格数。\n",
          "## 每家两个 mission（每个视频 = 同 init 的 pref 上排 / no-pref 下排）\n",
          "| 家 | preference | mission | 分岔 (pref vs naive) | 判据满足 pref/naive | 视频 |", "|---|---|---|---|---|---|"]
    picks = {}
    for r in rows:
        if r["ok"] and r["video"] != "-":
            picks.setdefault(r["fam"], []).append(r)
    for fam in sorted(FAMILY_NAME):
        ps = picks.get(fam, [])[:2]
        if not ps:
            md.append(f"| 3-{fam} | {FAMILY_NAME[fam]} | — | — | — | 无可用 mission |")
        for p in ps:
            dst = f"{fam:02d}_{FAMILY_NAME[fam]}__{p['mid']}.mp4"
            where = "approved" if os.path.exists(os.path.join(a.hr, "approved", dst)) else ""
            shutil.copyfile(os.path.join(a.hr, p["video"]), os.path.join(a.hr, where, dst))
            crit = f"{p['sat_p']:.0f}% / {p['sat_n']:.0f}%" if p["sat_p"] is not None and p["sat_n"] is not None else "-"
            md.append(f"| 3-{fam} | {p['pref_lan']} | {p['mid']} | {p['pref_branch']} vs {p['naive_branch']} | {crit} | {dst} |")
    md += ["\n## 全部 mission\n", "| mission | preference | mode | pref 分支 | naive 分支 | pref 生成 SR | naive SR(配对) | 判据满足 pref/naive | VLM 一致 | 状态 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = "OK" if r["ok"] else (r["note"] or "未跑完")
        crit = "-" if r["sat_p"] is None else f"{r['sat_p']:.0f}% / {'-' if r['sat_n'] is None else f'{r[chr(115)+chr(97)+chr(116)+chr(95)+chr(110)]:.0f}%'}"
        md.append(f"| {r['mid']} | {r['pref_lan']} | {r['mode']} | {r['pref_branch']} | {r['naive_branch']} | {r['pref_sr']} | {r['naive_sr']} | {crit} | {r['vlm'] or '-'} | {st} |")
    dec = os.path.join(a.hr, "_decisions.md")
    if os.path.exists(dec):
        md += ["", open(dec).read().rstrip()]
    os.makedirs(a.hr, exist_ok=True)
    open(os.path.join(a.hr, "REVIEW_NOTES.md"), "w").write("\n".join(md) + "\n")
    ok = sum(r["ok"] for r in rows); full = sum(1 for f in FAMILY_NAME if len(picks.get(f, [])) >= 2)
    print(f"{ok}/{len(rows)} missions OK, {full}/15 families have 2 -> {a.hr}/REVIEW_NOTES.md")
    for r in rows:
        if not r["ok"]: print(f"  !! {r['mid']}: {r['note'] or 'not finished'}")


if __name__ == "__main__":
    main()
