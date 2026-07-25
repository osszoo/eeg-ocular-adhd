"""Process-parallel driver for the stage-3 MODEL axis (PREREG 8.3.4).

Only the model axis is parallelised: it is the expensive half (~28 cells x 10
seeds x ~7 min = ~33 h single-threaded). The sample axis is Logistic-only and
cheap, so it stays serial via `ceiling.py --axis sample`.

WHY A SEPARATE FILE, NOT MORE FLAGS IN ceiling.py
  ceiling.py already separates computation (run_model_axis -> dict) from I/O
  (its drivers append CSVs). This wrapper reuses run_model_axis unchanged and
  adds ONLY parallel dispatch. No scoring logic is duplicated, so there is one
  source of truth for the numbers.

CONCURRENCY MODEL (chosen to make row corruption impossible)
  Workers COMPUTE ONLY and return a dict. The MAIN process is the only writer:
  it drains results via as_completed and appends both CSVs itself. No file lock,
  single scores.csv / selections.csv, no interleaved writes. This is the (c)
  option discussed in design: safer than per-worker files (no merge step) and
  simpler than file locking (no Windows lock portability issue).

RESUME
  On start the main reads _done_keys(scores.csv) and drops (cell, seed) pairs
  already present. Interrupt with Ctrl-C and rerun: only the remainder runs.
  Because the main is the sole writer, a row exists in scores.csv iff that unit
  fully finished, so resume never double-counts or half-writes.

THREADS
  Every fit is single-threaded (OMP etc. pinned to 1, inherited by children and
  re-pinned in worker_init). Parallelism is across cells/seeds, matching the
  cost model the timing probe used, so the ~7 min/cell constant still holds and
  wall time ~= 33 h / workers.

USAGE
  C:\\envs\\eeg\\python.exe exploration\\run_ceiling_parallel.py --seeds 10
  C:\\envs\\eeg\\python.exe exploration\\run_ceiling_parallel.py --seeds 10 --workers 24
  C:\\envs\\eeg\\python.exe exploration\\run_ceiling_parallel.py --seeds 10 --non-ocular-only
  C:\\envs\\eeg\\python.exe exploration\\run_ceiling_parallel.py --dry-run   # list units, write nothing
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import csv
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from classify import RANDOM_STATE                                  # noqa: E402
import ceiling                                                     # noqa: E402
from ceiling import (run_model_axis, model_grid, build_samples,    # noqa: E402
                     all_cells, cells_non_ocular, cells_ocular,
                     _done_keys, _ensure, _CEIL_DIR, _SCORES, _SELECT,
                     SCORE_HEADER, SEL_HEADER)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


def worker_init():
    """Re-pin single-thread in each child, belt and suspenders over inheritance."""
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'


# Module-level so it is picklable for ProcessPoolExecutor on Windows (spawn).
# _SAMPLES / _MODELS are filled once per worker via the initializer closure below.
_SAMPLES = None
_GRID = None
_MODELS = None


def _init_ctx(models_list):
    worker_init()
    global _SAMPLES, _GRID, _MODELS
    _SAMPLES = build_samples()
    _MODELS = set(models_list)
    _GRID = model_grid(_MODELS)


def _compute(unit):
    """Run one (sample, cond, seed) on the model axis. Pure compute; returns dict
    or None. Never writes."""
    sample, cond, seed = unit
    t0 = time.time()
    r = run_model_axis(sample, cond, _SAMPLES[sample], seed, _GRID, _MODELS)
    if r is None:
        return dict(unit=unit, ok=False, secs=time.time() - t0)
    return dict(unit=unit, ok=True, secs=time.time() - t0,
                auc=r['auc'], acc=r['acc'], sens=r['sens'], spec=r['spec'],
                n=r['n'], picks=r['picks'])


def _modal_triple(picks):
    triples = [(p[1], p[2], p[3]) for p in picks]
    return max(set(triples), key=triples.count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument('--models', nargs='+',
                    default=['logit', 'svm', 'rf', 'xgb', 'mlp'])
    ap.add_argument('--non-ocular-only', action='store_true')
    ap.add_argument('--ocular-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='list units and exit; write nothing')
    args = ap.parse_args()

    seeds = [RANDOM_STATE + i for i in range(args.seeds)]
    if args.ocular_only:
        cells = cells_ocular()
    elif args.non_ocular_only:
        cells = cells_non_ocular()
    else:
        cells = all_cells()

    units_all = [(s, c, sd) for sd in seeds for (s, c) in cells]

    _ensure(_SCORES, SCORE_HEADER)
    _ensure(_SELECT, SEL_HEADER)
    done = _done_keys(_SCORES, ['sample', 'cond', 'axis', 'outer_K', 'seed'])
    # model-axis rows carry axis='model', outer_K='5'
    todo = [u for u in units_all
            if (u[0], u[1], 'model', '5', str(u[2])) not in done]

    print('=' * 78)
    print('STAGE 3 MODEL AXIS -- parallel (PREREG 8.3.4)')
    print('=' * 78)
    print('  cells %d   seeds %d   units total %d   already done %d   to run %d'
          % (len(cells), len(seeds), len(units_all),
             len(units_all) - len(todo), len(todo)))
    print('  workers %d   models %s' % (args.workers, ','.join(args.models)))
    print('  writer: main process only (single scores/selections csv)')

    if args.dry_run:
        for u in todo[:12]:
            print('    would run:', u)
        if len(todo) > 12:
            print('    ... and %d more' % (len(todo) - 12))
        print('  dry run: nothing written.')
        return

    if not todo:
        print('  nothing to run; all units present in scores.csv.')
        return

    t0 = time.time()
    n_done = 0
    n_fail = 0
    secs_seen = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_ctx,
                             initargs=(args.models,)) as ex:
        futs = {ex.submit(_compute, u): u for u in todo}
        for fut in as_completed(futs):
            res = fut.result()
            unit = res['unit']
            s, c, sd = unit
            secs_seen.append(res['secs'])
            if not res['ok']:
                n_fail += 1
                print('[----] %-10s %-11s seed=%d no rows (%.0fs)'
                      % (s, c, sd, res['secs']))
                continue
            # MAIN is the only writer.
            with open(_SCORES, 'a', newline='', encoding='utf-8') as fh:
                bm, bl, bk = _modal_triple(res['picks'])
                csv.writer(fh).writerow([s, c, 'model', 5, sd, res['n'],
                    '%.4f' % res['auc'], '%.4f' % res['acc'],
                    '%.4f' % res['sens'], '%.4f' % res['spec'],
                    bm, bl, bk, '%.1f' % res['secs']])
            with open(_SELECT, 'a', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                for fold, model, label, kname in res['picks']:
                    w.writerow([s, c, sd, fold, model, label, kname])
            n_done += 1
            elapsed = time.time() - t0
            rate = elapsed / max(n_done, 1)
            remain = rate * (len(todo) - n_done) / 60.0
            print('[ok  ] %-10s %-11s seed=%d AUC=%.4f  (%.0fs)  '
                  '%d/%d done, ~%.0f min left'
                  % (s, c, sd, res['auc'], res['secs'],
                     n_done, len(todo), remain))
            sys.stdout.flush()

    wall = (time.time() - t0) / 60.0
    print('\n' + '-' * 78)
    print('  finished %d units (%d failed) in %.1f min wall'
          % (n_done, n_fail, wall))
    if secs_seen:
        import statistics as st
        print('  per-unit compute: median %.0fs  min %.0fs  max %.0fs'
              % (st.median(secs_seen), min(secs_seen), max(secs_seen)))
    print('  scores.csv and selections.csv appended.')
    print('  NEXT: sample axis (serial, cheap):')
    print('    C:\\\\envs\\\\eeg\\\\python.exe src\\\\ceiling.py --axis sample --seeds %d'
          % args.seeds)
    print('-' * 78)


if __name__ == '__main__':
    main()