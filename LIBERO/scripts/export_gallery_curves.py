#!/usr/bin/env python3
"""Re-evaluate gallery examples and export numerical curves without rendering.

These are selected-demo traces, not aggregate policy SR/SA. Missing tasks remain
missing; metric exceptions are recorded instead of being silently accepted.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

ROOT = Path(__file__).resolve().parents[2]


def numbers(values):
    import numpy as np
    return [float(v) if np.isfinite(v) else None for v in values]


def export_one(job):
    rec, out = job
    import render_demo_video as rdv
    import calibrate_low_pref as cal
    import numpy as np
    key = rec['suite'] + '/' + rec['task_id']
    result = dict(key=key, status=rec['status'], demo='demo_0', curves=[], errors=[],
                  interpretation='Selected demonstration only; not aggregate SR or SA.',
                  display_processing='Existing renderer: speed subtask interpolation and 3-frame median filter; raw arrays are also retained.')
    if rec['status'] == 'ok':
        cache = {}
        for ri, row in enumerate(rec['rows']):
            source = ROOT.parent / row['source']
            for mi, metric in enumerate(rec['metrics']):
                config = Path(metric.get('info') or cal._mission_info_path(metric['mission']))
                ck = (str(source), str(config))
                try:
                    if ck not in cache:
                        cache[ck] = rdv.replay_satisfaction(metric['mission'], str(source), 'demo_0', info_path=str(config), strict=True)
                    sat, raw, _, done, steps, _ = cache[ck]
                    mkey = {'max_acc': 'acc_range', 'avoid_object': 'avoid_to_object'}.get(metric.get('key'), metric.get('key'))
                    if mkey not in sat:
                        raise KeyError(f'Metric {mkey!r} absent from replay: {list(sat)}')
                    displayed, signal = sat[mkey], raw[mkey]
                    if len(displayed) != steps:
                        if len(displayed) % steps:
                            raise ValueError('Curve length is not a multiple of stored steps')
                        displayed = displayed[::len(displayed)//steps]
                        signal = signal[::len(signal)//steps]
                    finite = np.asarray(displayed)[np.isfinite(displayed)]
                    final = float(finite[-1]) if len(finite) else None
                    previous = next((x['satisfaction'] for x in rec['final'] if x['metric'] == metric['name'] and x['row'] == row['label']), None)
                    result['curves'].append(dict(row_index=ri, metric_index=mi, row=row['label'], metric=metric['name'],
                        source=row['source'], source_bytes=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                        config=str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config),
                        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), steps=steps,
                        stored_done_step=done, satisfaction=numbers(displayed), raw=numbers(signal),
                        final=final, rendered_final=previous,
                        endpoint_matches_video=(final is None and previous is None) or
                            (final is not None and previous is not None and abs(final-previous) < 1e-6)))
                except Exception as exc:
                    result['errors'].append(dict(row=row['label'], metric=metric['name'], error=f'{type(exc).__name__}: {exc}'))
        result['status'] = 'error' if result['errors'] else 'exported'
    target = Path(out) / (key + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, allow_nan=False))
    return dict(key=key, status=result['status'], curves=len(result['curves']), errors=result['errors'],
                endpoint_mismatches=sum(not c['endpoint_matches_video'] for c in result['curves']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    records = json.loads(args.manifest.read_text())
    if args.only:
        wanted = set(args.only.split(','))
        records = [r for r in records if r['suite']+'/'+r['task_id'] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(export_one, [(r, str(args.out)) for r in records]):
            results.append(result)
            (args.out / 'progress.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
