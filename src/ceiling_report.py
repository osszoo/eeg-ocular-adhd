"""Stage 3 report generator (PREREG 8.3.4 outputs).

READ-ONLY. Reads results/ceiling/scores.csv and selections.csv and prints four
tables. Runs no models; every number already exists in the CSVs. The combo arm
(table 4) is the one exception and is produced by a SEPARATE call with an
explicit cell (see --combo), because it needs a tuned-model x LOSO fit that the
star design did not run.

TABLES
  1  MODEL AXIS performance: per (cond, sample), median AUC over seeds with
     [min,max], plus acc/sens/spec medians. Rows in PRE-REGISTERED order
     (condition then sample), never sorted by AUC -- 8.3.4(10) forbids putting
     the maximum at the top.
  2  SAMPLE AXIS (learning curve): per (cond, sample), AUC at K=5 / K=10 / LOSO.
     K5/K10 are seed medians; LOSO is the single deterministic run. Reads whether
     more training subjects raise AUC (sample-size bottleneck) or not.
  3  GRID-BOUNDARY DIAGNOSTIC (8.3.4(7)): model and k selection frequencies,
     overall and per cell. FLAGS boundary concentration. k=all never chosen is
     GOOD (upper bound not binding); k=15 dominating is a LOWER-boundary caveat
     that must be recorded, since the grid stops at 15.
  4  COMBO ARM (exploratory, 8.3.4(9)): best signal x best model x LOSO, printed
     ONLY beside the null expected-max discount (8.3.3(4): 20 cfg +0.067, 50 cfg
     +0.082, 288 cfg +0.108). This is the sole table that carries a discount;
     tables 1-3 do not (8.3.4(10)).

WHAT THIS DELIBERATELY DOES NOT DO
  No "max + expected-max-under-null" framing on tables 1-3. Symmetric reporting
  of every cell, maximum not headlined. The discount rides only on table 4.

USAGE
  C:\\envs\\eeg\\python.exe src\\ceiling_report.py                 # tables 1-3
  C:\\envs\\eeg\\python.exe src\\ceiling_report.py --combo iii_noica tier2_86 svm
      # table 4: runs ONE tuned-model x LOSO fit for the named cell, appends a
      # combo row, prints it beside the discount. Choose the cell from table 1.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import csv
import argparse
import statistics as st
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import RANDOM_STATE                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_CEIL_DIR = _ROOT / 'results' / 'ceiling'
_SCORES = _CEIL_DIR / 'scores.csv'
_SELECT = _CEIL_DIR / 'selections.csv'

# Pre-registered display order (8.3.4(2)). Rows follow this, not AUC rank.
COND_ORDER = ['i_brain', 'iii_noica', 'iv_noasr', 'v_wideband',
              'ii_eye1', 'ii_eyeall']
SAMPLE_ORDER = ['full121', 'tier1_91', 'tier2_86', 'tact120',
                'roster84', 'roster59']

# Null expected-max discount, from PREREG section 1 (8.3.3(4)).
NULL_MAX = {20: 0.067, 50: 0.082, 288: 0.108}


def _read(path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _fmt_band(vals):
    if not vals:
        return '   --  '
    med = st.median(vals)
    return '%.3f [%.3f,%.3f]' % (med, min(vals), max(vals))


# ------------------------------------------------------------------- table 1
def table1(rows):
    print('=' * 92)
    print('TABLE 1  MODEL AXIS -- ceiling per cell (median over seeds [min,max])')
    print('=' * 92)
    print('  Rows in pre-registered order; NOT sorted by AUC (8.3.4(10)).')
    print('  %-11s %-10s %-22s %-7s %-7s %-7s  seeds' %
          ('cond', 'sample', 'AUC', 'acc', 'sens', 'spec'))
    print('  ' + '-' * 88)
    by = defaultdict(list)
    for r in rows:
        if r['axis'] != 'model':
            continue
        by[(r['cond'], r['sample'])].append(r)
    for cond in COND_ORDER:
        any_row = False
        for sample in SAMPLE_ORDER:
            rs = by.get((cond, sample))
            if not rs:
                continue
            any_row = True
            auc = [float(r['auc']) for r in rs]
            acc = [float(r['acc']) for r in rs]
            sen = [float(r['sens']) for r in rs]
            spc = [float(r['spec']) for r in rs]
            print('  %-11s %-10s %-22s %.3f   %.3f   %.3f   %2d'
                  % (cond, sample, _fmt_band(auc),
                     st.median(acc), st.median(sen), st.median(spc), len(auc)))
        if any_row:
            print('')


# ------------------------------------------------------------------- table 2
def table2(rows):
    print('=' * 92)
    print('TABLE 2  SAMPLE AXIS -- learning curve (K sweep, Logistic centre point)')
    print('=' * 92)
    print('  K5/K10: seed medians.  LOSO: single deterministic run.')
    print('  Rising with K => sample size is the bottleneck; flat => it is not.')
    print('  %-11s %-10s %-8s %-8s %-8s  %-8s' %
          ('cond', 'sample', 'K=5', 'K=10', 'LOSO', 'd(LOSO-K5)'))
    print('  ' + '-' * 74)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['axis'] != 'sample':
            continue
        by[(r['cond'], r['sample'])][r['outer_K']].append(float(r['auc']))
    for cond in COND_ORDER:
        any_row = False
        for sample in SAMPLE_ORDER:
            cell = by.get((cond, sample))
            if not cell:
                continue
            any_row = True
            k5 = st.median(cell['5']) if cell.get('5') else float('nan')
            k10 = st.median(cell['10']) if cell.get('10') else float('nan')
            lo = cell['loso'][0] if cell.get('loso') else float('nan')
            d = lo - k5 if not (np.isnan(lo) or np.isnan(k5)) else float('nan')
            print('  %-11s %-10s %.3f    %.3f    %.3f    %+.3f'
                  % (cond, sample, k5, k10, lo, d))
        if any_row:
            print('')


# ------------------------------------------------------------------- table 3
def table3(sel):
    print('=' * 92)
    print('TABLE 3  GRID-BOUNDARY DIAGNOSTIC (8.3.4(7))')
    print('=' * 92)
    if not sel:
        print('  selections.csv empty.')
        return
    models = Counter(r['model'] for r in sel)
    ks = Counter(r['k'] for r in sel)
    tot = len(sel)
    print('  overall model frequency (%d selections):' % tot)
    for m, c in models.most_common():
        print('    %-6s %4d  (%4.1f%%)' % (m, c, 100 * c / tot))
    print('  overall k frequency:')
    for k in ('15', '30', '60', 'all'):
        c = ks.get(k, 0)
        print('    k=%-4s %4d  (%4.1f%%)' % (k, c, 100 * c / tot))

    print('\n  boundary flags:')
    k_all = ks.get('all', 0)
    k_lo = ks.get('15', 0)
    if k_all == 0:
        print('    [OK]   k=all never chosen -> mRMR cap 60 is NOT binding'
              ' (upper bound clear).')
    else:
        print('    [WARN] k=all chosen %d times -> full 306 sometimes preferred;'
              ' upper region may bind.' % k_all)
    if k_lo > tot * 0.40:
        print('    [CAVEAT] k=15 is %.0f%% of picks -> LOWER boundary. Grid stops'
              ' at 15; true optimum may be < 15.' % (100 * k_lo / tot))
        print('             AUCs in table 1 are a LOWER bound under this grid'
              ' (PREREG 8.3.4(7); expand via k in {5,10} if pursued -- decision B).')
    # depth=None concentration (RF upper boundary)
    rf = [r for r in sel if r['model'] == 'rf']
    if rf:
        depth_none = sum(1 for r in rf if 'depth=None' in r['hp'])
        print('    RF depth=None share: %d/%d (%.0f%%)%s'
              % (depth_none, len(rf), 100 * depth_none / len(rf),
                 '  [WARN unbounded depth preferred]'
                 if depth_none > len(rf) * 0.6 else ''))

    print('\n  per-cell modal (model, k)  [most common pick across folds/seeds]:')
    per = defaultdict(list)
    for r in sel:
        per[(r['cond'], r['sample'])].append((r['model'], r['k']))
    for cond in COND_ORDER:
        for sample in SAMPLE_ORDER:
            picks = per.get((cond, sample))
            if not picks:
                continue
            mode = Counter(picks).most_common(1)[0]
            print('    %-11s %-10s  %-6s k=%-4s  (%d/%d folds)'
                  % (cond, sample, mode[0][0], mode[0][1], mode[1], len(picks)))


# ------------------------------------------------------------------- table 4
def table4_combo(cond, sample, model):
    """Runs ONE tuned-model x LOSO fit for the named cell and prints it beside the
    null discount. Separate from tables 1-3 because the star design never ran a
    tuned model under LOSO. Exploratory (8.3.4(9)); discount per 8.3.3(4)."""
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_sample_weight
    import ceiling as C

    print('=' * 92)
    print('TABLE 4  COMBO ARM (exploratory) -- %s x %s x %s x LOSO'
          % (cond, sample, model))
    print('=' * 92)
    print('  8.3.4(9): best signal x best model x LOSO, outside the star design.')
    print('  8.3.3(4): reported ONLY beside the null expected-max discount below.')

    samples = C.build_samples()
    if sample not in samples:
        print('  unknown sample'); return
    loaded = C.load_condition(cond, samples[sample])
    if loaded is None:
        print('  no rows'); return
    X, y, g, n_pool = loaded

    # inner tuning inside each LOSO fold would be very expensive; the combo arm
    # is exploratory, so we tune ONCE on a K=5 inner split to pick hp, then apply
    # that hp across LOSO. This is a deliberate simplification, stated as such.
    grid = C.model_grid({model})
    seed = RANDOM_STATE
    isgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr0, _ = next(iter(isgkf.split(np.zeros(len(y)), y, g)))
    key, _ = C.inner_select(X[tr0], y[tr0], g[tr0], n_pool, grid, seed)
    m_name, label, kname = key
    hp = C._hp_lookup(m_name, label)
    print('  tuned hp (one K=5 inner split): %s %s k=%s' % (m_name, label, kname))

    subs = list(dict.fromkeys(g.tolist()))
    pos = {s: i for i, s in enumerate(subs)}
    ysub = np.array([int(y[g == s][0]) for s in subs])
    tot = np.zeros(len(subs)); cnt = np.zeros(len(subs))
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(y)), y, g):
        p = C.refit_and_predict(X[tr], y[tr], g[tr], X[te], n_pool,
                                m_name, label, kname, hp)
        gte = g[te]
        for s in dict.fromkeys(gte.tolist()):
            mm = gte == s
            tot[pos[s]] += p[mm].mean(); cnt[pos[s]] += 1
    psub = tot / np.maximum(cnt, 1)
    auc = float(roc_auc_score(ysub, psub))
    print('\n  combo AUC = %.4f' % auc)
    print('\n  null expected-max discount (8.3.3(4)) -- subtract before believing:')
    for n_cfg, disc in sorted(NULL_MAX.items()):
        print('    if this is the max over ~%3d configs: expected-max-under-null'
              ' = %+.3f' % (n_cfg, disc))
    print('  -> the combo number is NOT a primary result; the pre-registered')
    print('     primary contrast concluded at stage 4 and is unchanged.')

    # append a combo row for the record
    _CEIL_DIR.mkdir(parents=True, exist_ok=True)
    combo_csv = _CEIL_DIR / 'combo.csv'
    new = not combo_csv.exists()
    with open(combo_csv, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['cond', 'sample', 'model', 'tuned_hp', 'k', 'auc'])
        w.writerow([cond, sample, m_name, label, kname, '%.4f' % auc])
    print('\n  appended to results/ceiling/combo.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combo', nargs=3, metavar=('COND', 'SAMPLE', 'MODEL'),
                    help='run table 4 for one cell, e.g. --combo iii_noica tier2_86 svm')
    args = ap.parse_args()

    if args.combo:
        table4_combo(*args.combo)
        return

    rows = _read(_SCORES)
    sel = _read(_SELECT)
    if not rows:
        print('scores.csv empty -- run ceiling.py first.'); return
    table1(rows)
    table2(rows)
    table3(sel)
    print('\n' + '=' * 92)
    print('  Tables 1-3 report every cell symmetrically; the maximum is not')
    print('  headlined (8.3.4(10)). For the combo arm (table 4) pick the top cell')
    print('  from table 1 and run:  ceiling_report.py --combo <cond> <sample> <model>')
    print('=' * 92)


if __name__ == '__main__':
    main()