#!/usr/bin/env python3
"""Read-only inventory of suite definitions and demonstration artifacts.

Writes a JSON evidence snapshot and Markdown tables. Never starts generation,
changes registries, or treats a retained successful demo count as raw rollout SR.
"""
import argparse
import csv
import datetime as dt
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'VLA_data'
REVIEW = Path('/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review')
RQS = {
    'SinglePref': ('Q1, Q2', 'RQ1, RQ2'),
    'PrefScale': ('Q3, Q4', 'RQ3'),
    'PrefCompose': ('Q2, Q5', 'RQ2, RQ4'),
    'PrefConflict': ('Q5', 'RQ5'),
    'ComprePref': ('Q1', 'RQ1'),
}
TARGETS = {'SinglePref': 100, 'PrefScale': 100, 'PrefCompose': 64, 'PrefConflict': 48}


def read_yaml(path):
    return yaml.safe_load((ROOT / path).read_text())


def path_text(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@lru_cache(None)
def inspect_hdf(path):
    p = Path(path)
    info = {'path': path_text(p), 'exists': p.exists(), 'episodes': 0,
            'done_positive': 0, 'done_missing': 0}
    if not p.exists():
        return info
    info['resolved_path'] = str(p.resolve())
    info['mtime'] = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat()
    try:
        with h5py.File(p, 'r') as f:
            data = f['data']
            demos = [v for k, v in data.items() if k.startswith('demo_') and isinstance(v, h5py.Group)]
            info['episodes'] = len(demos)
            for g in demos:
                if 'dones' not in g:
                    info['done_missing'] += 1
                else:
                    info['done_positive'] += int(np.any(g['dones'][...]))
    except (OSError, KeyError, ValueError) as exc:
        info['error'] = str(exc)
    return info


def hdf(folder, mid):
    return inspect_hdf(str(DATA / folder / f'Mission_{mid}.hdf5'))


@lru_cache(None)
def mission(mid):
    family, layout = mid.split('_', 1)
    series, pref = family.split('-')
    folder = ROOT / f'LIBERO/dataset/series_{series}/perm_{series}_{pref}/libero_{layout}'
    p = folder / 'mission_info.yaml'
    y = yaml.safe_load(p.read_text()) if p.exists() else {}
    bddl = folder / y.get('bddl_file', '__missing__')
    candidates = sorted(folder.glob('*.bddl'))
    if not bddl.exists() and len(candidates) == 1:
        bddl = candidates[0]
    return {'mission': mid, 'config': path_text(p), 'config_exists': p.exists(),
            'bddl': path_text(bddl) if bddl.exists() else '',
            'bddl_exists': bddl.exists(),
            'naive_bddls': [path_text(x) for x in sorted((folder / 'no_pref_layout').glob('*.bddl'))],
            'configured_task': y.get('language_instructions', ''),
            'preference_types': [x.get('type', '') for x in (y.get('low_prefs') or []) + (y.get('high_prefs') or [])]}


@lru_cache(None)
def log_evidence(mid):
    out = {'raw_sr': None, 'finish': [], 'metrics': [], 'videos': [], 'sheets': []}
    p = Path('/tmp/s3gen') / f'pref_{mid}.log'
    if p.exists():
        matches = re.findall(r'Mission\s+' + re.escape(mid) + r'\s+success rate:\s*(\d+)/(\d+)', p.read_text(errors='replace'))
        if matches:
            a, b = map(int, matches[-1])
            out['raw_sr'] = {'successes': a, 'attempts': b, 'path': str(p)}
    for folder in ('/tmp/s3gen', '/tmp/skillgen'):
        p = Path(folder) / f'finish_{mid}.json'
        if p.exists():
            try:
                out['finish'].append({'path': str(p), **json.loads(p.read_text())})
            except json.JSONDecodeError:
                out['finish'].append({'path': str(p), 'error': 'invalid JSON'})
        p = Path(folder) / f'table_{mid}.csv'
        if p.exists():
            with p.open() as f:
                out['metrics'].extend({'path': str(p), **r} for r in csv.DictReader(f))
    for s in REVIEW.glob('S*'):
        for p in (s / 'all').glob('*.mp4'):
            if re.search(r'(?:^|__)' + re.escape(mid) + r'(?:\.mp4|_)', p.name):
                out['videos'].append(str(p))
        p = s / 'sheets' / f'{mid}.json'
        if p.exists():
            out['sheets'].append(str(p))
    return out


def artifact(mid, suite):
    # Reused SinglePref missions retain their source; combo artifacts may be
    # generated into SinglePref by the current batch drivers.
    folders = ['PrefScale'] if suite == 'PrefScale' else ['SinglePref', suite]
    if mid.startswith('4-') and suite != 'PrefScale':
        folders = ['SinglePref', 'PrefCompose']
    candidates = [hdf(f, mid) for f in dict.fromkeys(folders)]
    existing = [x for x in candidates if x['exists'] and not x.get('error')]
    # Select the freshest artifact, not the largest older dataset.
    selected = max(existing, key=lambda x: x['mtime']) if existing else candidates[0]
    return {'selected': selected, 'candidates': candidates}


def build_rows():
    rows = []

    def add(suite, cell, condition, mids, task, prompt, registry, group):
        mids = [m.lower().replace('_ab', '_AB').replace('_a', '_A').replace('_b', '_B') for m in mids]
        sources = [{'definition': mission(m), 'data': artifact(m, suite),
                    'pref_pair': hdf('SinglePref_paired', m),
                    'naive_pair': hdf('SinglePref_naive_paired', m),
                    'naive': hdf('SinglePref_naive', m), 'evidence': log_evidence(m)} for m in mids]
        structure = all(s['definition']['config_exists'] and s['definition']['bddl_exists'] for s in sources)
        covered = all(s['data']['selected']['episodes'] > 0 for s in sources)
        ten = all(s['data']['selected']['episodes'] >= 10 for s in sources)
        hundred = all(s['data']['selected']['episodes'] >= 100 for s in sources)
        pair = all(s['pref_pair']['episodes'] >= 10 and s['naive_pair']['episodes'] >= 10 for s in sources)
        done_pair = pair and all(s['pref_pair']['done_positive'] >= 10 and s['naive_pair']['done_positive'] >= 10 for s in sources)
        flags = []
        for s in sources:
            m = s['definition']['mission']
            if not s['definition']['bddl_exists']:
                flags.append(f'{m}: missing BDDL/config')
            if not s['data']['selected']['episodes']:
                flags.append(f'{m}: no source episodes')
            sr = s['evidence']['raw_sr']
            if sr and sr['successes'] == 0:
                flags.append(f'{m}: latest raw generation {sr["successes"]}/{sr["attempts"]}')
            for f in s['evidence']['finish']:
                if f.get('ok') is False:
                    flags.append(f'{m}: finish failed at {f.get("stage", "unknown")}')
            for metric in s['evidence']['metrics']:
                try:
                    if float(metric['sat_pref']) <= float(metric['sat_naive']):
                        flags.append(f'{m}: recorded pref satisfaction <= naive')
                except (KeyError, TypeError, ValueError):
                    pass
        rows.append({'suite': suite, 'cell': cell, 'condition': condition, 'group': group,
                     'paper_q': RQS[suite][0], 'legacy_project_rq': RQS[suite][1],
                     'task': task, 'preference_prompt': prompt, 'registry': registry,
                     'sources': sources, 'structure_present': structure,
                     'data_source_present': covered, 'data_source_at_least_10': ten,
                     'data_source_at_least_100': hundred, 'paired_files_10_plus': pair,
                     'paired_done_markers_10_plus': done_pair,
                     'validated': 'not independently revalidated',
                     'human_reviewed': 'not established by this audit', 'flags': sorted(set(flags))})

    reg = 'LIBERO/dataset/perm_missions.yaml'
    d = read_yaml(reg)
    for family, layouts in d['pairings'].items():
        for layout in layouts:
            mid = f'{family}_{layout}'
            add('SinglePref', mid.lower(), 'P', [mid], d['tasks'][mid], d['preferences'][mid], reg, family)
    reg = 'VLA_data/PrefScale/tasksuite.yaml'
    d = read_yaml(reg)
    for mid, task in d['tasks'].items():
        add('PrefScale', mid, 'canonical', [mid], task, d['preferences'][mid], reg, mid.split('_')[0])
    reg = 'VLA_data/PrefCompose/tasksuite.yaml'
    d = read_yaml(reg)
    for cell, conds in d['conditions'].items():
        for cond, mid in conds.items():
            add('PrefCompose', cell + ':' + cond, cond, [mid], d['tasks'][mid], d['preferences'][mid], reg, cell)
    reg = 'VLA_data/PrefConflict/tasksuite.yaml'
    d = read_yaml(reg)
    for cell, mids in d['pp_pairs'].items():
        for i, mid in enumerate(mids, 1):
            add('PrefConflict', cell + f':P{i}', f'P{i}', [mid], d['tasks'][mid], d['preferences'][mid], reg, cell)
        add('PrefConflict', cell + ':P1+P2', 'P1+P2 (evaluation)', mids,
            d['tasks'][mids[0]], ' + '.join(str(d['preferences'][m]) for m in mids), reg, cell)
    for cell, val in d['pt_cells'].items():
        mid = val['task_A_c']
        for cond in ('task+P', 'task+null'):
            add('PrefConflict', cell + ':' + cond, cond, [mid], d['tasks'][mid],
                d['preferences'][mid] if cond == 'task+P' else '', reg, cell)
    reg = 'LIBERO/dataset/series_7/language.yaml'
    d = read_yaml(reg)
    for m in d['missions']:
        for setting in d['settings']:
            add('ComprePref', m['mission_id'] + ':' + setting, setting, [m['mission_id']],
                m['task'], m.get(setting, ''), reg, m['family'])
    return rows


def render(rows, out):
    summary = {}
    for suite in RQS:
        rs = [r for r in rows if r['suite'] == suite]
        summary[suite] = {'target': TARGETS.get(suite), 'registered_conditions': len(rs)}
        for key in ('structure_present', 'data_source_present', 'data_source_at_least_10',
                    'data_source_at_least_100', 'paired_files_10_plus', 'paired_done_markers_10_plus'):
            summary[suite][key] = sum(r[key] for r in rs)
        summary[suite]['unique_source_missions'] = len({s['definition']['mission'] for r in rs for s in r['sources']})
        summary[suite]['flagged_conditions'] = sum(bool(r['flags']) for r in rs)
        unique = {s['definition']['mission']: s for r in rs for s in r['sources']}
        summary[suite]['stored_source_episodes'] = sum(s['data']['selected']['episodes'] for s in unique.values())
        if suite == 'SinglePref':
            summary[suite]['episodes_missing_to_100_each'] = sum(max(0, 100 - s['data']['selected']['episodes']) for s in unique.values())
    snapshot = {'audited_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'summary': summary, 'rows': rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'suite_inventory_2026_09_05.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    def esc(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    text = ['---', 'title: Task, preference, suite and research-question evidence table',
            'type: findings', 'created: 2026-09-05', 'updated: 2026-09-05', 'status: audit', '---',
            '', '# Task–preference–suite–RQ evidence table', '',
            'This is a live filesystem inventory, not an acceptance certificate. Targets use the paper (§3.3): 100 / 100 / 48 / 64 = 312 base-suite conditions. ComprePref reuses tasks and is excluded from that sum.', '',
            '**RQ numbering:** the paper (§4, Q1–Q5) uses Q4 for generalization and Q5 for composition plus conflict. The older project experiment design uses RQ4 for composition and RQ5 for conflict. Both are shown below; they are not interchangeable.', '',
            'Sources: [paper](<../../../paper/1855_LIBERO_PeRM_Benchmarking_ (1).pdf>), [suite plan](../../../LIBERO/dataset/SUITE_PLAN_312.md), [legacy experiment design](../../experiments/experiment.md).', '',
            '## Summary', '',
            '| Suite | Paper Q | Old project RQ | Target | Registered | Config + BDDL | Source data present | Source >=10 | Source >=100 | Paired files >=10+10 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for suite, s in summary.items():
        text.append('| ' + ' | '.join(map(str, [suite, *RQS[suite], s['target'] or 'reuse',
                    s['registered_conditions'], s['structure_present'], s['data_source_present'],
                    s['data_source_at_least_10'], s['data_source_at_least_100'], s['paired_files_10_plus']])) + ' |')
    text += ['', '## Interpretation and limits', '',
             '- SinglePref membership comes from its 100 `pairings`, not all 115 task-text entries. Its inference tasksuite remains a separate 40-task configuration.',
             '- PrefScale currently registers 35 concrete missions; the remaining 65 target entries are not fabricated in this table. Its planned expansion and old layout directories are not a completed registry.',
             '- The old Scale registry requests 150 demos per listed mission, whereas the 312-task expansion plan proposes 50 demos for 88 source missions. Reconcile this target before generation; the table does not silently choose one.',
             '- Conflict counts 12 preference-pair layouts × 3 prompts + 6 task-conflict layouts × 2 prompts = 48 conditions. Joint conflict prompts reuse the two single-preference sources; they do not imply a successful jointly satisfiable demonstration. Task-conflict and null prompts share task-first source demonstrations.',
             '- `Source data present` means every required source has at least one retained HDF5 episode. Reused episodes are not counted as new physical data, and counts across suites must not be summed.',
             '- `Paired files` checks file counts only. JSON also records positive done markers. Neither verifies matching initial layouts or preference satisfaction. Automated finish flags and VLM sheets are evidence, not human approval.',
             '- Raw generation SR is parsed from the last `Mission ... success rate` summary, never from duplicated `HDF5 saved` log lines. Missing raw SR remains unknown. Stored demo counts are not raw rollout attempts.',
             '- Recorded calibration satisfaction is not automatically the paper SA: this audit does not recompute per-episode task-success-weighted discrete satisfaction.',
             '- HDF5 metadata and done arrays were read; trajectories were not replayed. Freshest source artifacts are selected, and all candidate locations are retained in the JSON snapshot.', '',
             'Full paths, raw SR, recorded metrics, automated finish results and review artifacts are in the [JSON evidence snapshot](suite_inventory_2026_09_05.json).', '']
    text += ['## Concrete next work', '',
             '- SinglePref: all 100 definitions have data, but only 36 have 100 stored pref episodes. The arithmetic gap to 100 each is 4,633 episodes across 64 missions, before quality rejection. Canonical paired files exist for 75 missions; the other 25 need artifact recovery or pairing. Six of those 75 have incomplete done markers, which requires diagnosis rather than declaring physical task failure.',
             '- Repair the recorded pairing failures for `3-1_115_1`, `3-5_120_1`, and `3-12_036_2`. Recheck the recorded motion preference/naive discrimination for `1-1_034` and `1-2_062` before accepting their selected pairs.',
             '- Compose: six conditions have no source episodes: `4-7_118:AB`, `4-10_070:AB`, `4-10_007:B`, `4-10_007:AB`, `4-11_058:B`, and `4-13_077:AB`. `4-3_116:AB` has only five. Thus seven conditions are below the ten-episode review threshold.',
             '- Conflict: 48 conditions have source data through 30 unique source missions (1,554 stored episodes); this does not establish joint conflict evaluation or human acceptance. Reused size/speed-source issues above also affect Conflict.',
             '- Scale: finish the 100-condition registry, reconcile its quota and source mapping, then restore/regenerate raw HDF5. Existing scene folders and LeRobot archives are not counted as verified raw HDF5 coverage.',
             '- ComprePref: 34 base missions × 2 rewrites = 68 instructions. Expanding the documented plan to 100 base missions requires another 66 missions / 132 rewritten instructions. SinglePref inference still lists 40 missions and must be reconciled with the new 100-task membership.',
             '- First fix acceptance reporting: a batch DONE marker or a video is not evidence that a zero-success condition passed. Recheck affected raw SR, selected paired artifacts and preference metrics before bulk generation.', '']
    for suite in RQS:
        text += ['## ' + suite, '',
                 '| Cell / condition | Task | Preference / instruction | Source missions | BDDL | Retained episodes | Raw SR | Paired pref/naive | Flags |',
                 '|---|---|---|---|---|---|---|---|---|']
        for r in (r for r in rows if r['suite'] == suite):
            ss = r['sources']
            values = [r['cell'], r['task'], r['preference_prompt'] or '(none)',
                      ', '.join(s['definition']['mission'] for s in ss),
                      '; '.join(s['definition']['bddl'] or 'MISSING' for s in ss),
                      '; '.join(str(s['data']['selected']['episodes']) for s in ss),
                      '; '.join(f'{s["evidence"]["raw_sr"]["successes"]}/{s["evidence"]["raw_sr"]["attempts"]}' if s['evidence']['raw_sr'] else 'unknown' for s in ss),
                      '; '.join(f'{s["pref_pair"]["episodes"]}/{s["naive_pair"]["episodes"]}' for s in ss),
                      '; '.join(r['flags']) or 'acceptance not established']
            text.append('| ' + ' | '.join(esc(v) for v in values) + ' |')
        text.append('')
    (out / 'task_preference_suite_rq_2026_09_05.md').write_text('\n'.join(text).rstrip() + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'ProjectProfile/Doc/findings')
    args = parser.parse_args()
    records = build_rows()
    counts = Counter(r['suite'] for r in records)
    assert counts['SinglePref'] == 100 and counts['PrefCompose'] == 64 and counts['PrefConflict'] == 48, counts
    assert len({(r['suite'], r['cell']) for r in records}) == len(records)
    print(json.dumps(render(records, args.out), indent=2))
