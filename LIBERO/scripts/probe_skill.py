#!/usr/bin/env python
"""单步执行一个 mission 的 plan, 把 mujoco 里的真实状态打出来 —— 调 skill 参数用。

不走 batch_run: 直接建 env、按 plan 逐 skill 跑, 每个 skill 边界记一次状态,
关键事件 (夹爪合拢 / 物体离地 / 下探最低点) 自动标注并存图。一次 ~30 秒。

用法:
  cd LIBERO && python scripts/probe_skill.py --mission 1-4_053 [--naive]
      [--offset pick_right_edge:white_bowl_1=0,0,0.02] [--plan task|naive]
      [--tag myrun] [--no-img]
  # 两组参数对比:
  python scripts/probe_skill.py --mission 1-4_053 --naive --tag A
  python scripts/probe_skill.py --mission 1-4_053 --tag B --diff A
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO)
OUT_DIR = "/data3/zhixu/PeRM_data/demos/_probe"

GRIP_CLOSED = 0.03      # m, 两指间距小于此视为合拢
LIFT_EPS = 0.01         # m, 物体比初始高出此值视为离地


def _mission_dir(mission_id):
    s, rest = mission_id.split("-", 1)
    grp, libero_id = rest.split("_", 1)
    return os.path.join(LIBERO, "dataset", f"series_{s}", f"perm_{s}_{grp}", f"libero_{libero_id}")


def _load_mission(mission_id):
    import yaml
    d = _mission_dir(mission_id)
    y = yaml.safe_load(open(os.path.join(d, "mission_info.yaml")))
    y["_dir"] = d
    y["_bddl"] = os.path.join(d, y["bddl_file"])
    return y


def _snapshot(env, watch_bodies):
    """当前 mujoco 状态的一行摘要。"""
    sim = env.sim
    ee = sim.data.get_body_xpos("gripper0_eef").copy()
    q = sim.data.get_body_xquat("gripper0_eef").copy()
    qpos = sim.data.qpos
    gid = [sim.model.joint_name2id(n) for n in sim.model.joint_names if "gripper0_finger" in n]
    sep = abs(qpos[sim.model.jnt_qposadr[gid[0]]] - qpos[sim.model.jnt_qposadr[gid[1]]]) if len(gid) >= 2 else float("nan")
    objs = {}
    for b in watch_bodies:
        try:
            objs[b] = sim.data.body_xpos[sim.model.body_name2id(b)].copy()
        except Exception:
            pass
    return {"ee": ee, "quat": q, "sep": float(sep), "objs": objs}


def _watch_bodies(env, task_objs):
    """plan 里出现过的物体 + 场景里所有 *_main body。"""
    names = [n for n in env.sim.model.body_names if n.endswith("_main")]
    return names[:12]


def run(mission_id, use_naive, offsets, tag, save_img=True, max_steps=1200, expand=True, seed=None):
    from libero.libero.envs import OffScreenRenderEnv
    from MotionGen.motion_gen import get_img_from_obs
    import MotionGen.motion_gen as mg

    y = _load_mission(mission_id)
    plan = y["naive_plan"] if use_naive else y["task_plan"]
    if expand:   # 复刻 TaskGen/task.py:112 —— 每个非 move_to 的 skill 前自动插 move_to
        plan = [step for sk, ob in plan for step in
                ([(sk, ob)] if sk.startswith("move_to") else [("move_to", ob), (sk, ob)])]
    sub = dict(y.get("subtask_offset") or {})
    for k, v in (offsets or {}).items():
        sub[k] = v
        print(f"  [override] {k} = {v}")

    env = OffScreenRenderEnv(bddl_file_name=y["_bddl"], camera_heights=256, camera_widths=256)
    if seed is not None:
        env.seed(seed)          # gen 每个 rollout 都 env.seed(batch_id), 初始 layout 随之变
    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))

    watch = _watch_bodies(env, plan)
    s0 = _snapshot(env, watch)
    init_z = {b: p[2] for b, p in s0["objs"].items()}
    print(f"\n== {mission_id}  plan={'naive' if use_naive else 'task'}  ({len(plan)} skills) ==")
    print(f"   起始 EE z={s0['ee'][2]:.3f}  gripper sep={s0['sep']:.4f}")

    trace, events, imgs = [], [], []
    t = 0
    from MotionGen.heuristic_skill import heuristic as H
    for si, (skill, obj) in enumerate(plan):
        obj_sim = mg.to_sim_env_name(obj) if obj else None
        off = sub.get(f"{skill}:{obj}") or sub.get("move_to:_all")
        off = off[0] if isinstance(off, list) and off and isinstance(off[0], list) else off
        try:
            acts = _skill_actions(env, H, skill, obj_sim, off)
        except Exception as e:
            print(f"  [{si}] {skill}({obj}) -> 生成动作失败: {e}")
            continue
        for a in acts:
            obs, _, done, _ = env.step(np.asarray(a, dtype=float))
            t += 1
            s = _snapshot(env, watch)
            trace.append({"t": t, "skill": skill, **{k: (v.tolist() if hasattr(v, "tolist") else v)
                                                     for k, v in s.items() if k != "objs"},
                          "objs": {b: p.tolist() for b, p in s["objs"].items()}})
            if t > max_steps:
                break
        s = _snapshot(env, watch)
        lifted = [b for b, p in s["objs"].items() if p[2] - init_z.get(b, p[2]) > LIFT_EPS]
        print(f"  [{si}] {skill:<18}{str(obj):<26} t={t:>4}  EE z={s['ee'][2]:.3f}  sep={s['sep']:.4f}"
              f"  {'离地:' + ','.join(b.replace('_main','') for b in lifted) if lifted else ''}")
        if save_img:
            imgs.append((f"{si}_{skill}", get_img_from_obs(obs, key="agentview_image"),
                         get_img_from_obs(obs, key="robot0_eye_in_hand_image")))
    return env, trace, imgs, watch, init_z


def _skill_actions(env, H, skill, obj, off):
    """把 skill 名映射到 heuristic 的动作序列。只覆盖 SinglePref 用到的那些。"""
    ANGLE = [np.pi, 0, -np.pi / 2]
    if skill == "lift_up":
        return H.move_at_z_axis(0.10, gripper_state=-1.0)
    if obj is None:
        return []
    xpos = env.sim.data.get_body_xpos(obj).copy()
    if skill in ("move_to", "move_to_straight", "move_to_away"):
        gs = -1.0
        if skill == "move_to_straight":
            return H.move_to_waypoint(env, target_pos=xpos, target_orn=ANGLE, gripper_state=gs, offset=off)
        return H.move_to(env, target_pos=xpos, target_orn=ANGLE, gripper_state=gs, offset=off)
    fn = {
        "pick": H.pick, "pick_right_edge": H.pick_right_edge, "pick_left_edge": H.pick_left_edge,
        "pick_back_edge": H.pick_back_edge, "pick_top_angle": H.pick_top_angle,
        "put": H.put, "put_right_edge": H.put_right_edge, "put_left_edge": H.put_left_edge,
    }.get(skill)
    if fn is None:
        raise ValueError(f"probe 还没接这个 skill: {skill}")
    kw = {"target_name": obj}
    if off is not None:
        kw["offset"] = off
    if skill == "pick":
        kw["yaml_offset"] = off is not None
    return fn(env, **kw)


def _events(trace, init_z):
    """从 trace 里找关键时刻: 首次合拢 / 下探最低点 / 物体首次离地。"""
    ev = {}
    seps = np.array([r["sep"] for r in trace])
    zs = np.array([r["ee"][2] for r in trace])
    closed = np.flatnonzero(seps < GRIP_CLOSED)
    if len(closed):
        ev["首次合拢"] = int(closed[0])
    ev["下探最低"] = int(np.argmin(zs))
    for b, z0 in init_z.items():
        col = np.array([r["objs"].get(b, [0, 0, z0])[2] for r in trace])
        up = np.flatnonzero(col - z0 > LIFT_EPS)
        if len(up):
            ev[f"{b.replace('_main','')} 离地"] = int(up[0])
    return ev


def _report(trace, events, watch, tag, imgs, mission_id, save_img):
    import cv2
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n   关键时刻 (t / EE 位姿 / 夹爪 / 与目标物的相对位置):")
    for name, t in sorted(events.items(), key=lambda kv: kv[1]):
        r = trace[min(t, len(trace) - 1)]
        ee = np.array(r["ee"])
        line = f"   {name:<22} t={t:>4}  EE=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:.3f})  sep={r['sep']:.4f}"
        near = sorted(((np.linalg.norm(ee - np.array(p)), b) for b, p in r["objs"].items()))[:1]
        if near:
            d, b = near[0]
            p = np.array(r["objs"][b])
            line += f"  最近物={b.replace('_main','')} dxy={np.linalg.norm((ee-p)[:2])*100:.1f}cm dz={(ee[2]-p[2])*100:+.1f}cm"
        print(line)
    js = os.path.join(OUT_DIR, f"{mission_id}_{tag}.json")
    json.dump({"mission": mission_id, "tag": tag, "events": events,
               "trace": trace[::max(1, len(trace) // 400)]}, open(js, "w"))
    print(f"\n   trace -> {js}")
    if save_img and imgs:
        rows = []
        for name, a, e in imgs:
            row = np.hstack([cv2.resize(a, (200, 200)), cv2.resize(e, (200, 200))])
            cv2.putText(row, name[:26], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            rows.append(row)
        grid = np.vstack(rows[:10])
        p = os.path.join(OUT_DIR, f"{mission_id}_{tag}.png")
        cv2.imwrite(p, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"   每个 skill 结束时的画面 -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--naive", action="store_true", help="跑 naive_plan (默认跑 task_plan)")
    ap.add_argument("--offset", action="append", default=[],
                    help="覆盖 subtask_offset, 如 pick_right_edge:white_bowl_1=0,0,0.02")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-img", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="env.seed(), 跟 gen 的 batch_id 对应")
    ap.add_argument("--raw", action="store_true", help="不自动插 move_to (默认插, 跟 gen 一致)")
    ap.add_argument("--diff", default=None, help="跟之前某个 tag 的 trace 对比关键时刻")
    a = ap.parse_args()
    offs = {}
    for o in a.offset:
        k, v = o.split("=", 1)
        offs[k] = [float(x) for x in v.split(",")]
    env, trace, imgs, watch, init_z = run(a.mission, a.naive, offs, a.tag, save_img=not a.no_img, expand=not a.raw, seed=a.seed)
    ev = _events(trace, init_z)
    _report(trace, ev, watch, a.tag, imgs, a.mission, not a.no_img)
    if a.diff:
        old = os.path.join(OUT_DIR, f"{a.mission}_{a.diff}.json")
        if os.path.exists(old):
            o = json.load(open(old))
            print(f"\n   vs [{a.diff}] 关键时刻对比:")
            for k in sorted(set(ev) | set(o["events"])):
                print(f"     {k:<22} 本次 t={ev.get(k,'-'):<6} {a.diff} t={o['events'].get(k,'-')}")
        else:
            print(f"   (没找到 {a.diff} 的 trace)")


if __name__ == "__main__":
    main()
