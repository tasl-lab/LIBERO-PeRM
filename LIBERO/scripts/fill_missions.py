#!/usr/bin/env python
"""Generate the smallest demo set a mission needs to be scored and plotted, in parallel.

Per mission:  with-preference demos (30 successes, 60 attempts)  ->  no-preference run on the
same initial states (10 successes within the first 30)  ->  init-paired 10 + 10 files
(pair_demos.py)  ->  anchors written to mission_info.yaml (calibrate_low_pref.py --apply).
If the no-preference run cannot reach 10 successes the mission stays with-preference only and is
anchored on its own demos. No-preference missions (--nopref) only get their own demos.

Never regenerates: a mission whose output file already exists (including a symlink to the
canonical copy, which h5py would otherwise overwrite in place) is left alone and the later
steps run on what is there.

Usage:
  cd LIBERO && python scripts/fill_missions.py --missions 1-1_004_1,2-5_116_2 [--nopref 0-1_032_1,...] \
      [--jobs 10] [--logdir /data3/zhixu/PeRM_data/logs/fill]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
V = os.path.join(os.path.dirname(LIBERO), "VLA_data")
PY = sys.executable
# Each generator process would otherwise start ~200 BLAS / OpenMP threads; on a shared machine a dozen of
# them push the load far past the core count without running any faster.
ENV = dict(os.environ, MUJOCO_GL="egl", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
           MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", OMP_WAIT_POLICY="passive")
PREF = ("--tasksuite SinglePref --naive_llm_plan 0 --run_layout_gen 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 "
        "--generate_evaluation_data 0 --verbose --enable_collision_check 1 --batch_start 0 --manually_set_offset 1 "
        "--check_init_layout 0 --gen_in_batch 1 --overwrite_old 1 --use_forcibly 1").split()
NAIVE = ("--tasksuite SinglePref --check_init_layout 0 --run_layout_gen 0 --generate_evaluation_data 0 --gen_in_batch 1 "
         "--naive_llm_plan 0 --run_naive_plan 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 --overwrite_old 1 "
         "--use_forcibly 1 --gen_naive_anchor --force_naive_plan 1").split()


def ndemos(path):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return len(f["data"])
    except Exception:
        return 0


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=LIBERO, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def last_sr(log):
    try:
        lines = [l for l in open(log, errors="ignore") if "success rate" in l]
        return lines[-1].strip().split("success rate")[-1].strip(" :") if lines else None
    except OSError:
        return None


def fill(job):
    mid, nopref, a = job
    t0 = time.time(); r = dict(mission=mid, kind="nopref" if nopref else "pref")
    L = lambda s: os.path.join(a.logdir, f"{s}_{mid}.log")
    P, N = f"{V}/SinglePref/Mission_{mid}.hdf5", f"{V}/SinglePref_naive/Mission_{mid}.hdf5"
    if os.path.lexists(P):
        r["pref_gen"] = "kept existing"
    else:
        tgt, att = (a.nopref_target, a.nopref_target * 3) if nopref else (a.pref_target, a.max_att)
        run([PY, "main.py", "--mission_id", mid] + PREF + ["--batch_size", str(tgt), "--max_attempts", str(att)], L("pref"))
        r["pref_gen"] = f"SR {last_sr(L('pref'))}"
    r["pref_demos"] = ndemos(P)
    if nopref or r["pref_demos"] == 0:
        r.update(ok=r["pref_demos"] >= (a.nopref_target if nopref else 1), minutes=round((time.time() - t0) / 60, 1))
        return r
    if os.path.lexists(N) and not os.path.exists(L("naive")):
        r["naive_gen"] = "kept existing (no log: cannot re-pair)"
    else:
        if not os.path.lexists(N) and not os.path.exists(L("naive")):      # a log without a file = already tried
            run([PY, "main.py", "--mission_id", mid] + NAIVE + ["--pair_init_from", P, "--batch_size", str(a.naive_target),
                 "--max_attempts", str(min(r["pref_demos"], a.pref_target))], L("naive"))
        r["naive_demos"] = ndemos(N)
        if r["naive_demos"] >= a.naive_target:
            rc = run([PY, "scripts/pair_demos.py", mid, L("naive"), "--n", str(a.naive_target)], L("pair"))
            line = [l for l in open(L("pair"), errors="ignore") if l.startswith("PAIR ")]
            r["pair"] = json.loads(line[-1][5:]) if line else {"ok": False, "note": "pair step crashed"}
            if rc == 0:
                rc = run([PY, "scripts/calibrate_low_pref.py", "--suite", "single-pref", "--pref_dir", f"{V}/SinglePref_paired",
                          "--naive_dir", f"{V}/SinglePref_naive_paired", "--missions", mid, "--n_demos", str(a.naive_target), "--apply"], L("calib"))
                r["calibrated"] = rc == 0
        else:
            # the no-preference plan does not work from these initial states (e.g. a neighbouring object
            # blocks the plain grasp): keep the mission with-preference only, anchored on its own demos
            rc = run([PY, "scripts/calibrate_low_pref.py", "--suite", "single-pref", "--pref_dir", f"{V}/SinglePref",
                      "--missions", mid, "--n_demos", str(a.naive_target), "--apply"], L("calib"))
            r.update(pref_only=True, calibrated=rc == 0)
    r["ok"] = bool(r.get("calibrated"))
    r["minutes"] = round((time.time() - t0) / 60, 1)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default=""); ap.add_argument("--nopref", default="")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--pref_target", type=int, default=30); ap.add_argument("--max_att", type=int, default=60)
    ap.add_argument("--naive_target", type=int, default=10); ap.add_argument("--nopref_target", type=int, default=10)
    ap.add_argument("--logdir", default="/data3/zhixu/PeRM_data/logs/fill")
    ap.add_argument("--out", default=None, help="status json (default <logdir>/status_<timestamp>.json)")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    jobs = [(m, False, a) for m in a.missions.split(",") if m] + [(m, True, a) for m in a.nopref.split(",") if m]
    out = a.out or os.path.join(a.logdir, f"status_{time.strftime('%m%d_%H%M%S')}.json")
    res = []
    with ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(fill, jobs):
            res.append(r); json.dump(res, open(out, "w"), indent=1)
            print(("OK   " if r["ok"] else "FAIL ") + json.dumps(r), flush=True)
    print(f"done: {sum(r['ok'] for r in res)}/{len(res)} ok -> {out}", flush=True)
