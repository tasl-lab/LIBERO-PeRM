#!/usr/bin/env python
"""
Unified batch runner for P_Bench missions.

Usage:
  python batch_run/batch_run.py                    # dev:      run/generate from dev_mission_ids
  python batch_run/batch_run.py --mode test        # test:     run/generate from test_mission_ids
  python batch_run/batch_run.py --mode data_gen    # data_gen: formal data generation from data_gen_mission_ids (videos: first 3 only)

Behavior depends on `parallel_execution` in batch_run.yaml:
  - True:  run missions in parallel directly (ThreadPoolExecutor)
  - False: generate a .sh script (batch_run.sh or test_before_push.sh)
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import yaml

WORKDIR = Path(__file__).resolve().parent.parent  # LIBERO/
YAML_PATH = Path(__file__).resolve().parent / "batch_run.yaml"

DEFAULT_MAIN_ARGS = [
    "--naive_llm_plan", "0",
    "--run_layout_gen", "0",
    "--high_pref_gpt_plan", "0",
    "--low_pref_gpt_plan", "0",
    "--generate_evaluation_data", "0",
    "--verbose",
    "--enable_collision_check", "1",
    "--batch_start", "0",
    "--manually_set_offset", "1",
    "--check_init_layout", "0",
    "--gen_in_batch", "1",
]


def load_config(path: str | None = None) -> dict:
    with open(path or YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Parallel execution
# ─────────────────────────────────────────────

SR_PATTERN = re.compile(r"Mission (\S+) success rate: (\d+/\d+)")


class LiveDashboard:
    """ANSI-based live dashboard: one fixed line per mission + progress bar."""

    def __init__(self, mission_ids: List[str]):
        self._lock = threading.Lock()
        self._mission_ids = list(mission_ids)
        # Per-mission state: status emoji, SR text
        self._status: dict[str, str] = {mid: "⏳ pending" for mid in mission_ids}
        self._lines = len(mission_ids) + 1  # missions + progress bar
        self._progress = ""
        self._drawn = False

    def _render(self):
        """Redraw the entire dashboard in place."""
        if self._drawn:
            # Move cursor up to overwrite previous dashboard
            sys.stdout.write(f"\033[{self._lines}A")

        for mid in self._mission_ids:
            sys.stdout.write(f"\033[K  {mid:<16} {self._status[mid]}\n")
        sys.stdout.write(f"\033[K{self._progress}\n")
        sys.stdout.flush()
        self._drawn = True

    def update_sr(self, mission_id: str, sr_text: str):
        with self._lock:
            self._status[mission_id] = f"🔄 SR: {sr_text}"
            self._render()

    def mark_running(self, mission_id: str):
        with self._lock:
            self._status[mission_id] = "🔄 running..."
            self._render()

    def mark_done(self, mission_id: str, success: bool, detail: str = ""):
        with self._lock:
            if not success:
                self._status[mission_id] = f"❌ FAIL  {detail}"
            else:
                # Parse SR from detail like "2/4  SR=0.50  AvgSteps=..."
                sr_match = re.search(r"SR=([\d.]+)", detail)
                sr_val = float(sr_match.group(1)) if sr_match else 1.0
                if sr_val < 0.2:
                    icon = "❌"
                elif sr_val < 0.5:
                    icon = "⚠️"
                else:
                    icon = "✅"
                self._status[mission_id] = f"{icon} {detail}"
            self._render()

    def update_progress(self, completed: int, total: int, failed_count: int):
        filled = int(30 * completed / max(total, 1))
        bar = "#" * filled + "-" * (30 - filled)
        percent = (completed / max(total, 1)) * 100
        self._progress = f"  Progress [{bar}] {completed}/{total} ({percent:5.1f}%) | failed: {failed_count}"
        # Don't render here — caller should render via other update methods

    def initial_draw(self):
        with self._lock:
            self._progress = f"  Progress [{'─' * 30}] 0/{len(self._mission_ids)} (  0.0%) | failed: 0"
            self._render()


def run_one_mission(mission_id: str, batch_size: int, overwrite_old: int,
                    use_forcibly: int, log_dir: Path,
                    dashboard: LiveDashboard,
                    tasksuite: str | None = None,
                    n_save_video: int | None = None,
                    max_attempts: int | None = None,
                    save_eval_hdf5: bool = False,
                    save_gif: bool = False,
                    gen_naive_anchor: bool = False) -> tuple[str, int]:
    dashboard.mark_running(mission_id)
    cmd = [
        sys.executable, "main.py",
        "--mission_id", mission_id,
        "--batch_size", str(batch_size),
        "--overwrite_old", str(overwrite_old),
        "--use_forcibly", str(use_forcibly),
        *DEFAULT_MAIN_ARGS,
    ]
    if tasksuite:
        cmd += ["--tasksuite", tasksuite]
    if n_save_video is not None:
        cmd += ["--n_save_video", str(n_save_video)]
    if max_attempts is not None:
        cmd += ["--max_attempts", str(max_attempts)]
    if save_eval_hdf5:
        cmd += ["--save_eval_hdf5"]
    if save_gif:
        cmd += ["--save_gif"]
    if gen_naive_anchor:
        cmd += ["--gen_naive_anchor"]
    log_path = log_dir / f"{mission_id}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        env_unbuf = os.environ.copy()
        env_unbuf["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=WORKDIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env_unbuf,
        )
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            m = SR_PATTERN.search(line)
            if m:
                dashboard.update_sr(mission_id, m.group(2))
        proc.wait()
    return mission_id, proc.returncode


def run_parallel(mission_ids: List[str], batch_size: int, overwrite_old: int,
                 use_forcibly: int, workers: int, mode: str,
                 tasksuite: str | None = None,
                 n_save_video: int | None = None,
                 max_attempts: int | None = None,
                 save_eval_hdf5: bool = False,
                 save_gif: bool = False,
                 gen_naive_anchor: bool = False) -> int:
    log_dir = WORKDIR / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    total = len(mission_ids)
    print(f"Mode: {mode} | Parallel: {workers} workers | {total} missions")
    print(f"Logs: {log_dir}")
    print("")

    dashboard = LiveDashboard(mission_ids)
    dashboard.initial_draw()

    failed = []
    start_time = time.time()
    sr_file = WORKDIR / "batch_run" / "sr_summary.txt"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one_mission, mid, batch_size, overwrite_old,
                            use_forcibly, log_dir, dashboard, tasksuite, n_save_video, max_attempts, save_eval_hdf5, save_gif, gen_naive_anchor)
            for mid in mission_ids
        ]
        for done_count, future in enumerate(as_completed(futures), start=1):
            mission_id, returncode = future.result()
            dashboard.update_progress(done_count, total, len(failed) + (1 if returncode != 0 else 0))
            if returncode != 0:
                failed.append(mission_id)
                dashboard.mark_done(mission_id, False, f"-> {log_dir / f'{mission_id}.log'}")
            else:
                sr_info = get_mission_sr(sr_file, mission_id)
                dashboard.mark_done(mission_id, True, sr_info)

    elapsed = time.time() - start_time
    print("")

    # ── SR Summary ──
    sr_file = WORKDIR / "batch_run" / "sr_summary.txt"
    sr_results = parse_sr_summary(sr_file, mission_ids)

    bar = "══" * 24
    print(bar)
    print(f"  {total - len(failed)} passed, {len(failed)} failed / {total} total  ({elapsed:.1f}s)")
    print(bar)

    if sr_results:
        print("")
        print(f"  {'Mission':<20} {'SR':>10} {'Avg Steps':>12}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 12}")
        for mid, sr, avg_steps in sr_results:
            print(f"  {mid:<20} {sr:>10} {avg_steps:>12}")
        print("")

    if failed:
        print("  Failed missions:")
        for mid in failed:
            print(f"    - {mid}")
        return 1

    # 必须每个 mission 都拿到绿 ✅（sr >= 0.5）才算 all pass
    non_green = []
    for mid, sr_str, _ in sr_results:
        m = re.search(r"SR=([\d.]+)", sr_str)
        sr_val = float(m.group(1)) if m else -1.0
        if sr_val < 0.5:
            non_green.append((mid, sr_str))

    if non_green:
        print("  Missions with warning / red X (SR < 0.5):")
        for mid, sr_str in non_green:
            print(f"    - {mid}  {sr_str}")
        return 1

    if mode == "test":
        print("  All missions passed! Safe to push.")
    return 0


def get_mission_sr(sr_file: Path, mission_id: str) -> str:
    """Get the latest SR line for a mission from sr_summary.txt."""
    if not sr_file.exists():
        return "SR: N/A"
    text = sr_file.read_text(encoding="utf-8")
    # Find all lines for this mission, take the last one
    sr_line = ""
    avg_steps = ""
    for i, line in enumerate(text.split("\n")):
        if line.startswith(mission_id + "\t"):
            sr_line = line.split("\t", 1)[1].replace("\t", "  ")
            # Check next line for Avg Steps
            lines = text.split("\n")
            if i + 1 < len(lines) and "Avg Steps" in lines[i + 1]:
                m = re.search(r"Avg Steps:\s*([\d.]+)\s*\((\d+) episodes\)", lines[i + 1])
                if m:
                    avg_steps = f"  AvgSteps={m.group(1)} ({m.group(2)}ep)"
    return (sr_line + avg_steps) if sr_line else "SR: N/A"


def parse_sr_summary(sr_file: Path, mission_ids: List[str]) -> list[tuple[str, str, str]]:
    """Parse sr_summary.txt, return last entry per mission in mission_ids order."""
    if not sr_file.exists():
        return []

    text = sr_file.read_text(encoding="utf-8")
    # Each block: "2-12_085\t2/2\tSR=1.00\n  Avg Steps: 172.0 (2 episodes)\n──────"
    sr_map: dict[str, tuple[str, str]] = {}
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match: mission_id \t succ/total \t SR=x.xx
        m = re.match(r"^(\S+)\t(\d+/\d+\tSR=[\d.]+)", line)
        if m:
            mid = m.group(1)
            sr = m.group(2).replace("\t", "  ")
            avg_steps = ""
            if i + 1 < len(lines) and "Avg Steps" in lines[i + 1]:
                avg_match = re.search(r"Avg Steps:\s*([\d.]+)", lines[i + 1])
                if avg_match:
                    avg_steps = avg_match.group(1)
                i += 1
            sr_map[mid] = (sr, avg_steps)
        i += 1

    results = []
    for mid in mission_ids:
        if mid in sr_map:
            sr, avg = sr_map[mid]
            results.append((mid, sr, avg))
        else:
            results.append((mid, "N/A", "N/A"))
    return results


# ─────────────────────────────────────────────
# Sequential script generation
# ─────────────────────────────────────────────

def _dev_command_block(mission_id: str, batch_size: int,
                       overwrite_old: int, use_forcibly: int,
                       tasksuite: str | None = None,
                       n_save_video: int | None = None,
                       max_attempts: int | None = None) -> str:
    label = f"Mission {mission_id}"
    padded = label.center(48)
    bar = "══" * 24
    tasksuite_arg = f"  --tasksuite {tasksuite} \\\n" if tasksuite else ""
    nsv_arg = f"  --n_save_video {n_save_video} \\\n" if n_save_video is not None else ""
    max_arg = f"  --max_attempts {max_attempts} \\\n" if max_attempts is not None else ""
    return f"""echo ""
echo "🚀🚀🚀 {bar} 🚀🚀🚀"
echo "🚀🚀🚀 {padded} 🚀🚀🚀"
echo "🚀🚀🚀 {bar} 🚀🚀🚀"
echo ""
python main.py \\
  --mission_id "{mission_id}" \\
  --naive_llm_plan 0 \\
  --run_layout_gen 0 \\
  --high_pref_gpt_plan 0 \\
  --low_pref_gpt_plan 0 \\
  --generate_evaluation_data 0 \\
  --verbose \\
  --enable_collision_check 1 \\
  --batch_size {batch_size} \\
  --batch_start 0 \\
  --manually_set_offset 1 \\
  --check_init_layout 0 \\
  --gen_in_batch 1 \\
  --overwrite_old {overwrite_old} \\
{tasksuite_arg}{nsv_arg}{max_arg}  --use_forcibly {use_forcibly} || true

"""


def generate_dev_script(mission_ids: List[str], batch_size: int,
                        overwrite_old: int, use_forcibly: int,
                        tasksuite: str | None = None,
                        n_save_video: int | None = None,
                        max_attempts: int | None = None) -> str:
    script_path = str(WORKDIR / "batch_run.sh")
    sr_log = "batch_run/sr_summary.txt"
    bar = "══" * 24

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\nset -e\n\n# Auto-generated from batch_run.yaml\n\n")
        f.write(f'> dataset/total_SR_log.txt\n> {sr_log}\n\n')

        for mid in mission_ids:
            f.write(_dev_command_block(str(mid), batch_size, overwrite_old, use_forcibly, tasksuite, n_save_video, max_attempts))

        f.write(f'\n# ── SR Summary ──\n')
        f.write(f'echo ""\n')
        f.write(f'echo "🏁🏁🏁 {bar} 🏁🏁🏁"\n')
        f.write(f'echo "🏁🏁🏁 {"Data Gen SR Summary".center(48)} 🏁🏁🏁"\n')
        f.write(f'echo "🏁🏁🏁 {bar} 🏁🏁🏁"\n')
        f.write(f'echo ""\n')
        f.write(f'if [ -s {sr_log} ]; then\n  cat {sr_log}\nelse\n')
        f.write(f'  echo "  (no SR data recorded)"\nfi\necho ""\n')

    os.chmod(script_path, 0o755)
    return script_path


def generate_test_script(mission_ids: List[str], batch_size: int,
                         overwrite_old: int, use_forcibly: int,
                         tasksuite: str | None = None) -> str:
    script_path = str(WORKDIR / "test_before_push.sh")
    sr_log = "batch_run/sr_summary.txt"
    total = len(mission_ids)
    bar = "══" * 24

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n# Auto-generated from batch_run.yaml (test_mission_ids)\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\ncd "$SCRIPT_DIR"\n\n')
        f.write('PASS_COUNT=0\nFAIL_COUNT=0\nFAIL_LIST=""\n\n')
        f.write(f'> dataset/total_SR_log.txt\n> {sr_log}\n\n')
        f.write(f'echo "{bar}"\n')
        f.write(f'echo "  test_before_push: {total} missions to verify"\n')
        f.write(f'echo "{bar}"\necho ""\n\n')

        for i, mid in enumerate(mission_ids, 1):
            mid_str = str(mid)
            f.write(f'echo "────────────────────────────────────────────────"\n')
            f.write(f'echo "  [{i}/{total}] Mission {mid_str}"\n')
            f.write(f'echo "────────────────────────────────────────────────"\n')
            f.write(f'if python main.py \\\n')
            f.write(f'  --mission_id "{mid_str}" \\\n')
            f.write(f'  --naive_llm_plan 0 \\\n')
            f.write(f'  --run_layout_gen 0 \\\n')
            f.write(f'  --high_pref_gpt_plan 0 \\\n')
            f.write(f'  --low_pref_gpt_plan 0 \\\n')
            f.write(f'  --generate_evaluation_data 0 \\\n')
            f.write(f'  --verbose \\\n')
            f.write(f'  --enable_collision_check 1 \\\n')
            f.write(f'  --batch_size {batch_size} \\\n')
            f.write(f'  --batch_start 0 \\\n')
            f.write(f'  --manually_set_offset 1 \\\n')
            f.write(f'  --check_init_layout 0 \\\n')
            f.write(f'  --gen_in_batch 1 \\\n')
            f.write(f'  --overwrite_old {overwrite_old} \\\n')
            if tasksuite:
                f.write(f'  --tasksuite {tasksuite} \\\n')
            f.write(f'  --use_forcibly {use_forcibly}; then\n')
            f.write(f'  echo "  ✅ PASS"\n  PASS_COUNT=$((PASS_COUNT + 1))\n')
            f.write(f'else\n  echo "  ❌ FAIL"\n  FAIL_COUNT=$((FAIL_COUNT + 1))\n')
            f.write(f'  FAIL_LIST="$FAIL_LIST {mid_str}"\nfi\necho ""\n\n')

        f.write(f'echo "{bar}"\n')
        f.write(f'echo "  Test Summary: $PASS_COUNT passed, $FAIL_COUNT failed / {total} total"\n')
        f.write(f'echo "{bar}"\n')
        f.write(f'if [ $FAIL_COUNT -gt 0 ]; then\n')
        f.write(f'  echo "  Failed missions:$FAIL_LIST"\n  exit 1\n')
        f.write(f'else\n  echo "  All missions passed! Safe to push."\n  exit 0\nfi\n')

    os.chmod(script_path, 0o755)
    return script_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P_Bench batch runner")
    parser.add_argument("--config", default=None,
                        help="alternate batch_run yaml (default: batch_run/batch_run.yaml)")
    parser.add_argument("--mode", choices=["dev", "test", "data_gen"], default=None,
                        help="Override yaml `mode`. dev: dev_mission_ids | test: test_mission_ids | "
                             "data_gen: data_gen_mission_ids (videos for first 3 only)")
    args = parser.parse_args()

    cfg = load_config(getattr(args, "config", None))

    mode = args.mode or cfg.get("mode", "dev")
    if mode not in ("dev", "test", "data_gen"):
        print(f"Invalid mode in yaml: {mode!r} (must be dev/test/data_gen)", file=sys.stderr)
        return 1

    id_key = {"dev": "dev_mission_ids", "test": "test_mission_ids", "data_gen": "data_gen_mission_ids"}[mode]
    mission_ids: List[str] = [str(m) for m in (cfg.get(id_key) or [])]
    size_key = {"dev": "dev_batch_size", "test": "test_batch_size", "data_gen": "data_gen_batch_size"}[mode]
    default_size = {"dev": 5, "test": 5, "data_gen": 50}[mode]
    batch_size = int(cfg.get(size_key) or default_size)
    overwrite_old = int(cfg.get("overwrite_old", 0))
    use_forcibly = int(cfg.get("use_forcibly", 0))
    parallel = bool(cfg.get("parallel_execution", False))
    workers = int(cfg.get("workers", 4))
    tasksuite = cfg.get("tasksuite") or None
    n_save_video = 3 if mode == "data_gen" else None
    # data_gen: target-success loop until batch_size 个 success 或 max_attempts 次尝试。dev/test 保留原固定次数语义（max_attempts=None）。
    max_attempts = int(cfg.get("data_gen_max_attempts", 100)) if mode == "data_gen" else None
    save_eval_hdf5 = bool(cfg.get("save_eval_hdf5", False))
    save_gif = bool(cfg.get("save_gif", False))
    gen_naive_anchor = bool(cfg.get("gen_naive_anchor", False))
    if gen_naive_anchor and not parallel:
        print("gen_naive_anchor requires parallel_execution: true — the non-parallel "
              "script path does not emit --gen_naive_anchor (flag would be silently "
              "ignored, producing normal pref demos).", file=sys.stderr)
        return 1

    if not mission_ids:
        print(f"No {id_key} found in {YAML_PATH}", file=sys.stderr)
        return 1

    print(f"Mode: {mode} | Source: {id_key} | {len(mission_ids)} missions | "
          f"batch_size={batch_size} | parallel={parallel}"
          + (f" | n_save_video={n_save_video}" if n_save_video is not None else "")
          + (f" | max_attempts={max_attempts}" if max_attempts is not None else ""))

    if parallel:
        return run_parallel(mission_ids, batch_size, overwrite_old,
                            use_forcibly, workers, mode, tasksuite, n_save_video, max_attempts, save_eval_hdf5, save_gif, gen_naive_anchor)
    else:
        if mode == "test":
            script = generate_test_script(mission_ids, batch_size, overwrite_old, use_forcibly, tasksuite)
        else:
            script = generate_dev_script(mission_ids, batch_size, overwrite_old, use_forcibly, tasksuite, n_save_video, max_attempts)
        print(f"Generated: {script}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())