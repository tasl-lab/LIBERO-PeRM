#!/usr/bin/env python
"""Series-2 (skill-pref) human-review summary: one row per mission, one pick per family.

Reads what the finish pipeline left behind (/tmp/skillgen/finish_<id>.json, table_<id>.csv,
the pref-gen SR lines in /tmp/skillgen_all.log + batch_run/sr_summary.txt) and writes
  <HR>/REVIEW_NOTES.md            31-row table + per-family pick
  <HR>/<family>__<mission>.mp4    the 14 per-family videos (copied from <HR>/all/)
Usage: cd LIBERO && python scripts/skill_review_table.py [--hr DIR]
"""
import argparse, csv, glob, json, os, re, shutil, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
HR_DEFAULT = "/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2"
LOGD = "/tmp/skillgen"
STAND_IN = {"2-5_116_1": 6}
FAMILY_NAME = {1: "pick_edge(left)", 2: "pick_edge(back)", 3: "pick_handle", 4: "pick_top_angle", 5: "pick_from_side",
               6: "drawer+pick_from_side", 7: "pick_horizontal", 8: "push", 9: "drag", 10: "throw",
               11: "pick_from_front", 12: "put_with_rotation", 13: "pick_edge(right)", 14: "lay_down"}


def missions():
    out = []
    for d in sorted(glob.glob(os.path.join(LIBERO, "dataset/series_2/perm_2_*/libero_*"))):
        lid = os.path.basename(d)[len("libero_"):]
        if "_nopref" in lid or (re.match(r"116_\d+$", lid) and lid != "116_1"):   # layout-variant scratch dirs; 116_1 kept as the 2-6 stand-in candidate
            continue
        fam = int(re.search(r"perm_2_(\d+)", d).group(1))
        mid = f"2-{fam}_{lid}"
        fam = STAND_IN.get(mid, fam)   # 2-5_116_1 stands in for the 2-6 family (decision 2026-09-01)
        out.append((fam, mid, d))
    return sorted(out, key=lambda x: (x[0], x[1]))


def skills(plan):
    return "+".join(s[0] for s in (plan or []) if isinstance(s, (list, tuple)) and s and s[0] != "move_to")


def pref_gen_sr(mid):
    """pref-demo generation SR: today's phase-2 line if any, else the last 100-demo run in sr_summary."""
    try:
        for line in open("/tmp/skillgen_all.log"):
            m = re.match(rf"pref\s+{re.escape(mid)} -> success rate: (\d+/\d+)", line)
            if m: return m.group(1)
    except FileNotFoundError:
        pass
    for lf in (f"{LOGD}/pref_{mid}.log",):
        if os.path.exists(lf):
            hits = re.findall(r"success rate: (\d+/\d+)", open(lf, errors="ignore").read())
            if hits: return hits[-1]
    last = None
    try:
        for line in open(os.path.join(LIBERO, "batch_run/sr_summary.txt"), errors="ignore"):
            m = re.match(rf"{re.escape(mid)}\t(\d+)/(\d+)", line)
            if m and int(m.group(1)) >= 30: last = f"{m.group(1)}/{m.group(2)} (earlier run)"
    except FileNotFoundError:
        pass
    return last or "?"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hr", default=HR_DEFAULT); a = ap.parse_args()
    rows = []
    for fam, mid, d in missions():
        y = yaml.safe_load(open(os.path.join(d, "mission_info.yaml"))) or {}
        lp = (y.get("low_prefs") or [{}])[0]
        r = {"fam": fam, "mid": mid, "pref_skill": skills(y.get("task_plan")), "naive_skill": skills(y.get("naive_plan")),
             "metric": lp.get("type", "-"), "pref_sr": pref_gen_sr(mid), "gate": "-", "naive_sr": "-", "P": None, "N": None,
             "d": None, "ok": False, "note": ""}
        jf = f"{LOGD}/finish_{mid}.json"
        if os.path.exists(jf):
            j = json.load(open(jf)); r["ok"] = bool(j.get("ok")); r["gate"] = j.get("gate", "-") or "-"
            r["naive_sr"] = j.get("naive_sr", "-") or "-"; r["note"] = "" if r["ok"] else f"{j.get('stage')}: {j.get('note')}"
        # rescoring runs (SKIP_GEN) lose the gate / naive counts; recover them from the batch driver logs
        if r["gate"] in ("-", "") or r["naive_sr"] in ("-", "", "10/0"):
            for lf in sorted(glob.glob("/tmp/skillgen_finish_*.log")):
                for line in open(lf, errors="ignore"):
                    m = re.match(rf"{re.escape(mid)} OK\s+gate=(\S+) naive=(\S+)", line.strip())
                    if m and m.group(1) and m.group(2) not in ("10/0", ""):
                        r["gate"], r["naive_sr"] = m.group(1), m.group(2)
        tf = f"{LOGD}/table_{mid}.csv"
        r["sat_p"] = r["sat_n"] = r["vlm"] = None
        if os.path.exists(tf):
            for c in csv.DictReader(open(tf)):
                fl = lambda v: float(v) if v not in ("", None) else None
                r["P"] = fl(c["mean_pref"]); r["N"] = fl(c["mean_naive"]); r["d"] = fl(c["cohen_d"])
                r["sat_p"] = fl(c.get("sat_pref")); r["sat_n"] = fl(c.get("sat_naive"))
        # VLM double-check: sheets/<mid>.json holds the metric verdicts per demo, sheets/<mid>.vlm.json my calls
        sj, vj = os.path.join(a.hr, "sheets", f"{mid}.json"), os.path.join(a.hr, "sheets", f"{mid}.vlm.json")
        if os.path.exists(sj) and os.path.exists(vj):
            m, v = json.load(open(sj)), json.load(open(vj))
            agree, tot = 0, 0
            for side in ("pref", "naive"):
                for x, call in zip(m.get(side, []), v.get(side, [])):
                    if x.get("verdict") is None: continue
                    tot += 1; agree += int((x["verdict"] >= 100) == bool(call))
            r["vlm"] = f"{agree}/{tot}" if tot else None
        r["video"] = "-"
        for cand in (f"all/{mid}_pref_vs_nopref.mp4", f"all/{mid}_demo_0.mp4"):
            if os.path.exists(os.path.join(a.hr, cand)):
                r["video"] = cand; break
        rows.append(r)
    f = lambda v, n=4: "-" if v is None else f"{v:.{n}g}"
    md = ["# Skill-preference（series_2）待审：14 家 × 2 mission = 28 个视频\n",
          "每个 mission：pref demo 先过 gate（naive 中位数 + 3 稳健 σ），naive 按同一批 init 逐条配对，再剔掉判据不满足的 pref demo，"
          "最终 pref / naive 各 10 条同 init。打分是节点式离散 0/100（判据见文末），P/N = 配对集上的均值，cohen_d 用 10 vs 10 算。\n"
          "「判据满足」= 10 条 pref / 10 条 naive 里判 100 的比例；「VLM 一致」= 我按 sheets/<mission>.png 逐格看图判的结果与 metric 一致的格数。\n",
          "## 每家两个 mission（每个视频 = 同 init 的 pref 上排 / no-pref 下排；按 |cohen_d| 排）\n", "| 家 | preference | mission | metric | P | N | cohen_d | 视频 |", "|---|---|---|---|---|---|---|---|"]
    # two missions per family (the deliverable is 14 x 2 videos), ranked by |cohen_d|
    picks = {}
    for r in rows:
        if not r["ok"] or r["video"] == "-":
            continue
        picks.setdefault(r["fam"], []).append(r)
    for fam in sorted(FAMILY_NAME):
        ps = sorted(picks.get(fam, []), key=lambda r: -(abs(r["d"]) if r["d"] is not None else -1.0))[:2]
        if not ps:
            md.append(f"| 2-{fam} | {FAMILY_NAME[fam]} | — | — | — | — | — | 无可用 mission |")
        for p in ps:
            dst = f"{fam:02d}_{FAMILY_NAME[fam]}__{p['mid']}.mp4"
            # a family the user already signed off lives in approved/ (moved by
            # ~/Zhixu_Agent/approve_review.sh): refresh it there, don't resurrect it on top
            where = "approved" if os.path.exists(os.path.join(a.hr, "approved", dst)) else ""
            shutil.copyfile(os.path.join(a.hr, p["video"]), os.path.join(a.hr, where, dst))
            md.append(f"| 2-{fam} | {FAMILY_NAME[fam]} | {p['mid']} | {p['metric']} | {f(p['P'])} | {f(p['N'])} | {f(p['d'],3) if p['d'] is not None else 'pref-only'} | {dst} |")
    picks = {fam: ps[0] for fam, ps in picks.items() if ps}
    md += ["\n## 全部 mission\n", "| mission | pref skill | naive skill | metric | pref 生成 SR | gate 通过 | naive SR(配对) | P | N | cohen_d | 判据满足 pref/naive | VLM 一致 | 状态 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = "OK" if r["ok"] else (r["note"] or "未跑完")
        crit = "-" if r.get("sat_p") is None else f"{r['sat_p']:.0f}% / {'-' if r.get('sat_n') is None else f'{r[chr(115)+chr(97)+chr(116)+chr(95)+chr(110)]:.0f}%'}"
        md.append(f"| {r['mid']} | {r['pref_skill']} | {r['naive_skill']} | {r['metric']} | {r['pref_sr']} | {r['gate']} | {r['naive_sr']} | {f(r['P'])} | {f(r['N'])} | {f(r['d'],3)} | {crit} | {r.get('vlm') or '-'} | {st} |")
    dec = os.path.join(a.hr, "_decisions.md")
    if os.path.exists(dec):
        md += ["", open(dec).read().rstrip()]
    os.makedirs(a.hr, exist_ok=True)
    open(os.path.join(a.hr, "REVIEW_NOTES.md"), "w").write("\n".join(md) + "\n")
    ok = sum(r["ok"] for r in rows); print(f"{ok}/{len(rows)} missions OK, {len(picks)}/14 families covered -> {a.hr}/REVIEW_NOTES.md")
    for r in rows:
        if not r["ok"]: print(f"  !! {r['mid']}: {r['note'] or 'not finished'}")


if __name__ == "__main__":
    main()
