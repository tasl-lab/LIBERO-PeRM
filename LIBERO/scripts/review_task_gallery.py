#!/usr/bin/env python
"""Check every rendered task clip against what its condition should look like and
collect the ones that do not into <gallery>/needs_review/<reason>/ for a human pass.

Expectation per condition (high = end value >= 70, low = end value <= 30):
  single / scale / conflict P1,P2 : every metric high on the with-preference rollout, low on a
                                    without-preference / no-preference rollout
  conflict P1+P2                  : each rollout high on its own metric, low on the other
  conflict task+P                 : task-following rollout low, preference-following high
  conflict task+null              : task-following rollout low
  compose nopref / A / B / AB     : metric A high iff A is requested, same for B

A curve that swings by 40+ points three or more times is flagged as well, whatever its end value.
Tasks whose demonstrations could not be generated at all get their failure recordings copied into
needs_review/0_demo_cannot_be_generated/.

A metric that has no calibration anchors (and is not a node criterion) is flagged first: it returns a
constant, so its curve would pass the end-value checks without meaning anything.

Writes needs_review/REVIEW.md + overview.jpg and tags each manifest entry with `review`.

Usage:  python scripts/review_task_gallery.py [--gallery /data3/zhixu/PeRM_data/demos/_task_gallery]
"""
import argparse, collections, json, os, shutil, subprocess

import yaml

HI, LO = 70, 30
DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
NOGEN = "0_demo_cannot_be_generated"
NOCAL = "0_metric_not_calibrated"
CAL_KEY = {"max_acc": "acc_range", "avoid_object": "avoid_to_object"}    # low_prefs type -> calibration key
REASONS = collections.OrderedDict([
    (NOCAL,                        "metric 没有标定锚点（分数是常数或没意义，曲线不能信）"),
    ("1_curve_never_fires",        "曲线全程没出分（metric 的判定节点没触发）"),
    ("2_pref_demo_scores_low",     "带 preference 的 demo 自己都没拿到高分"),
    ("3_composed_demo_misses_one", "A+B 的 demo 有一个 preference 没满足"),
    ("4_conflict_not_exclusive",   "冲突的两个 preference，metric 分不开（两边都高）"),
    ("5_high_without_preference",  "没给 preference 的 rollout 也拿了高分"),
    ("8_fewer_than_10_demos",      "生成成功率太低，凑不够 10 条 demo（曲线是用仅有的几条画的）"),
    ("6_curve_oscillates",         "曲线中途在高低之间来回大跳（终值可能是对的，但过程不像一条正常的满意度曲线）"),
    ("7_speed_baseline_high",      "speed 类：没被要求 speed 的那条 rollout，speed 分数没落到低位"),
])
ADVICE = {
    NOCAL: "直接补标定：用这个 mission 的 pref demo 和同场景的 no-pref demo 跑一遍 calibrate_low_pref.py --apply，再重渲染。",
    NOGEN: "看失败录像定：多半是抓取 / 放置偏移没对上，或者旁边的物体挡了路。要么调这个 mission 的 plan / offset，要么换 layout。",
    "1_curve_never_fires":        "倾向改 metric：节点没触发时该给 0 还是不计分要定一个；4-13 是抓取 metric 绑死在没被抓的那个物体上，要改成跟着实际抓的物体走。",
    "2_pref_demo_scores_low":     "倾向重标定：判据阈值 / 方向对这个 mission 写反了，demo 本身看着是对的。",
    "3_composed_demo_misses_one": "倾向重新生成 demo：A+B 的生成没把 B 做到位（或者 B 的锚点是别的 layout 标的）。",
    "4_conflict_not_exclusive":   "倾向改 metric：现在的判据两种做法都满足，需要一个能区分两者的量。",
    "5_high_without_preference":  "分情况：no-pref 路径本来就满足（要改 layout 或重生成 no-pref demo），或者 metric 没标定。",
    "8_fewer_than_10_demos":      "倾向换 layout 或调抓取：这个 preference 在这个场景里成功率只有个位数，补到 10 条要跑很多次。",
    "6_curve_oscillates":         "倾向改 metric 的 running 算法：acc 是对不断变长的样本取 P95，早期样本少就会来回跳；终值不受影响，只影响过程曲线。",
    "7_speed_baseline_high":      "双锚已经上了，剩下的不是公式问题：要么另一个 preference 顺带改了步数（从左边抓少走十来步，从前面抓多走几步），而这个场景里 Speed up 的 demo 只比 no-pref 快 11%，锚差太小压不住；要么 demo 只有 3 条、标不出 no-pref 锚。倾向重新生成：把 Speed up 的 demo 生成得更快（锚差拉到 25% 以上），或者把 demo 补到 10 条。",
}


def mission_dir(mid):
    fam, lay = mid.split("_", 1); a_, b_ = fam.split("-")
    return os.path.join(DATASET, f"series_{a_}", f"perm_{a_}_{b_}", f"libero_{lay}")


def uncalibrated(m):
    """True when a metric is scored against demonstration anchors (it has no node criterion) and the
    mission it is read from carries no anchor for it. Such a metric returns a constant, so a clip that
    looks fine (100 on the with-preference rollout) would pass every end-value check below."""
    if not m.get("type") or not m.get("mission"):
        return False
    path = m.get("info") or os.path.join(mission_dir(m["mission"]), "mission_info.yaml")
    y = yaml.safe_load(open(path)) or {}
    lp = next((l for l in y.get("low_prefs") or [] if l.get("type") == m["type"]), None)
    if lp is None or "criterion_threshold" in (lp.get("params") or {}):
        return False
    c = (y.get("low_pref_calibration") or {}).get(CAL_KEY.get(m["type"], m["type"])) or {}
    return c.get("pref_anchor") is None


def check(r):
    """Return list of (reason, message) for one rendered manifest entry."""
    names = [m["name"] for m in r["metrics"]]; types = [m.get("type") for m in r["metrics"]]
    nm, nr = len(names), len(r["rows"])
    order = [(mi, ri) for ri in range(nr) for mi in range(nm)]
    v = {k: x["satisfaction"] for k, x in zip(order, r["final"])}
    s, c = r["suite"], r.get("condition")
    out = [(NOCAL, f"「{m['name']}」取自 {m['mission']}，那里没有锚点") for m in r["metrics"] if uncalibrated(m)]

    def want(mi, ri, high, what):
        x = v.get((mi, ri))
        if x is None:
            out.append(("1_curve_never_fires", f"{what}：没出分")); return
        if high and x < HI:
            reason = "3_composed_demo_misses_one" if ((s == "compose" and c == "AB") or (s == "scale" and nm > 1)) else "2_pref_demo_scores_low"
            out.append((reason, f"{what}：应该高，实际 {x:.0f}"))
        if not high and x > LO:
            if types[mi] == "speed_range":
                reason = "7_speed_baseline_high"
            elif s == "conflict" and c == "P1+P2":
                reason = "4_conflict_not_exclusive"
            else:
                reason = "5_high_without_preference"
            out.append((reason, f"{what}：应该低，实际 {x:.0f}"))

    if s in ("single", "scale") or (s == "conflict" and c in ("P1", "P2")):
        for mi in range(nm):
            for ri, row in enumerate(r["rows"]):
                with_pref = row["label"].startswith("WITH ")       # not "WITHOUT ..."
                who = ("带 preference 的 rollout" if with_pref else "no-pref rollout") + (f"的「{names[mi]}」" if nm > 1 else "")
                want(mi, ri, with_pref, who)
    elif s == "conflict" and c == "P1+P2":
        want(0, 0, True, f"上排对「{names[0]}」"); want(1, 1, True, f"下排对「{names[1]}」")
        want(1, 0, False, f"上排对「{names[1]}」"); want(0, 1, False, f"下排对「{names[0]}」")
    elif s == "conflict" and c == "task+P":
        want(0, 0, False, "照 task 做的 rollout"); want(0, 1, True, "照 preference 做的 rollout")
    elif s == "conflict" and c == "task+null":
        want(0, 0, False, "照 task 做的 rollout")
    elif s == "compose":
        exp = {"nopref": (False, False), "A": (True, False), "B": (False, True), "AB": (True, True)}[c]
        for mi in (0, 1):
            want(mi, 0, exp[mi], f"{c} 条件下的「{names[mi]}」")
    few = [f"{row['label'].lower()} 只有 {row['n_demos']} 条" for row in r["rows"] if row.get("n_demos", 10) < 10]
    if few:
        out.append(("8_fewer_than_10_demos", "；".join(few)))
    for x in r["final"]:
        if x.get("swings", 0) >= 3:
            out.append(("6_curve_oscillates", f"「{x['metric']}」在 {x['row'].lower()} 上来回摆了 {x['swings']} 次（每次 ≥40 分）"))
    return out


def overview(groups, rev, cols=4, tw=480, th=240, bar=26):
    """One image with the last frame of every flagged clip, grouped by reason: needs_review/overview.jpg."""
    from PIL import Image, ImageDraw
    items = [(reason, r) for reason in REASONS for r in groups.get(reason, [])]
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + bar)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (reason, r) in enumerate(items):
        x, y = (i % cols) * tw, (i // cols) * (th + bar)
        d.rectangle([x, y, x + tw - 1, y + bar - 1], fill=(29, 29, 31))
        d.text((x + 6, y + 7), f"{reason[:28]} | {r['suite']}/{r['task_id']}", fill="white")
        sheet.paste(Image.open(os.path.join(rev, reason, f"{r['suite']}__{r['task_id']}.jpg")).resize((tw, th)), (x, y + bar))
    sheet.save(os.path.join(rev, "overview.jpg"), quality=88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="/data3/zhixu/PeRM_data/demos/_task_gallery")
    a = ap.parse_args()
    mpath = os.path.join(a.gallery, "manifest.json")
    man = json.load(open(mpath))
    rev = os.path.join(a.gallery, "needs_review")
    shutil.rmtree(rev, ignore_errors=True)
    groups = collections.defaultdict(list)
    for r in man:
        r.pop("review", None)
        if r["status"] != "ok":
            continue
        issues = check(r)
        if not issues:
            continue
        reason = sorted({i[0] for i in issues})[0]            # most serious reason decides the folder
        r["review"] = dict(reason=reason, issues=[i[1] for i in issues])
        groups[reason].append(r)
        d = os.path.join(rev, reason); os.makedirs(d, exist_ok=True)
        for ext in (".mp4", ".jpg"):
            shutil.copy2(os.path.join(a.gallery, r["suite"], r["task_id"] + ext), os.path.join(d, f"{r['suite']}__{r['task_id']}{ext}"))
    json.dump(man, open(mpath, "w"), indent=1)

    # tasks with no demonstrations although generation was tried: bring their failure recordings along
    nogen = []
    for r in man:
        mid = r.get("mission")
        if r["status"] != "missing" or not mid or "(" in mid:
            continue
        fd = os.path.join(mission_dir(mid), "Videos", "fail_case")
        vids = sorted(v for v in (os.listdir(fd) if os.path.isdir(fd) else []) if v.endswith(".mp4"))[:2]
        if not vids:
            continue
        d = os.path.join(rev, NOGEN); os.makedirs(d, exist_ok=True)
        for v in vids:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(fd, v), "-c:v", "libx264", "-crf", "26",
                            "-pix_fmt", "yuv420p", os.path.join(d, f"{r['suite']}__{r['task_id']}__fail_{v}")])
        nogen.append(r)

    ok = [r for r in man if r["status"] == "ok"]
    L = ["# 需要你过目的曲线", "",
         f"共渲染 {len(ok)} 个 task，其中 {sum(len(v) for v in groups.values())} 个曲线跟预期对不上，按原因分在下面的子目录里（视频 + 末帧截图）。",
         f"判定口径：终值 ≥ {HI} 算高，≤ {LO} 算低；每个条件该高还是该低见 `LIBERO/scripts/review_task_gallery.py` 开头。", ""]
    if nogen:
        L += [f"## {NOGEN}（{len(nogen)} 个）", "", "试过生成，成功数是 0，所以没有 demo 也没有曲线。目录里是失败录像（每个 2 条）。", "",
              f"我的建议：{ADVICE[NOGEN]}", "", "| suite | task | mission | preference |", "|---|---|---|---|"]
        L += [f"| {r['suite']} | `{r['task_id']}` | `{r['mission']}` | {r.get('title', '')} |" for r in nogen] + [""]
    for reason, title in REASONS.items():
        rs = groups.get(reason, [])
        if not rs:
            continue
        L += [f"## {reason}（{len(rs)} 个）", "", f"{title}。", "", f"我的建议：{ADVICE[reason]}", "",
              "| suite | task | preference | 哪里不对 |", "|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['suite']} | `{r['task_id']}` | {r['title']} | {'；'.join(r['review']['issues'])} |")
        L.append("")
    miss = collections.Counter((r["suite"], r.get("reason")) for r in man if r["status"] == "missing")
    L += ["## 还没法渲染的 task", "", "| suite | 数量 | 原因 |", "|---|---|---|"]
    L += [f"| {s} | {n} | {why} |" for (s, why), n in sorted(miss.items())]
    os.makedirs(rev, exist_ok=True)
    open(os.path.join(rev, "REVIEW.md"), "w").write("\n".join(L) + "\n")
    overview(groups, rev)
    print(f"rendered {len(ok)}, flagged {sum(len(v) for v in groups.values())}, cannot generate {len(nogen)}:")
    if nogen:
        print(f"  {NOGEN:<30}{len(nogen):>3}  " + " ".join(f"{r['suite'][:4]}/{r['task_id']}" for r in nogen))
    for reason in REASONS:
        if groups.get(reason):
            print(f"  {reason:<30}{len(groups[reason]):>3}  " + " ".join(f"{r['suite'][:4]}/{r['task_id']}" for r in groups[reason]))
