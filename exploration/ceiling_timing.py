"""Stage 3 timing probe -- wall-clock only, deliberately reports no score.

WHY THIS EXISTS
  Stage 3 multiplies the stage 4 cost by (inner folds x grid size x models).
  Whether that is two hours or three days decides how many seeds PREREG 8.3.4
  can register. This measures the constant instead of guessing it.

WHAT IT DELIBERATELY DOES NOT DO
  It never prints AUC or any other score. PREREG 8.3.4 is not committed yet, so
  seeing performance now would break the registration order. Models are fitted
  and predict_proba is called only so the clock is honest; the output is
  discarded unread.

WHAT IT MEASURES
  One (cell, seed, outer fold, inner fold) block -- the unit the full run
  repeats seeds x 5 x 3 times per cell. Cost is linear in that count, so the
  block time extrapolates exactly.

THREADS
  Every BLAS / OpenMP pool is pinned to 1 before numpy is imported. The full run
  will parallelise over folds and grid points, not inside a fit, so the
  single-thread constant is the one that extrapolates cleanly.

WEIGHTING CHECK (step 0)
  sklearn 1.9 exposes sample_weight on MLPClassifier.fit, which removes the
  asymmetry that would otherwise force MLP to run unweighted. Step 0 verifies
  (a) balanced sample weights reproduce class_weight='balanced' exactly on the
  stage 4 estimator, and (b) sample_weight survives a Pipeline into MLP. If (a)
  fails, stage 3 cannot claim to extend the stage 4 estimator, so it is checked
  before anything is timed.

NOTE ON SELECTION
  mRMR is fitted once at k=60 and prefixes of its support are used for the
  smaller k. Only column COUNT affects timing, so column identity is irrelevant
  here; the order-preserving selector that stage 3 needs belongs in the stage 3
  code, after PREREG 8.3.4.

USAGE
  C:\\envs\\eeg\\python.exe exploration/ceiling_timing.py
  C:\\envs\\eeg\\python.exe exploration/ceiling_timing.py --sample full121 --cond i_brain
  C:\\envs\\eeg\\python.exe exploration/ceiling_timing.py --models mlp svm
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                     # noqa: E402
from classify_ext import MRMR                            # noqa: E402
from contrast_ic import build_samples, load_condition    # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

# Timing probe only: convergence noise would bury the numbers being measured.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

K_INNER = 3
K_MAX = 60
K_GRID = [15, 30, 60, None]          # None = no selection, all pool columns

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


# ----------------------------------------------------------------- candidate grids
def grid_logit():
    return [('C=%g' % c, (lambda c=c: LogisticRegression(
        penalty='l2', C=c, max_iter=1000, random_state=RANDOM_STATE)))
        for c in (0.1, 1.0, 10.0)]


def grid_svm():
    out = []
    for c in (1.0, 10.0):
        for gm in ('scale', 0.01):
            out.append(('C=%g,g=%s' % (c, gm), (lambda c=c, gm=gm: SVC(
                kernel='rbf', C=c, gamma=gm, probability=True,
                random_state=RANDOM_STATE))))
    return out


def grid_rf():
    out = []
    for d in (None, 8):
        for leaf in (1, 5):
            out.append(('depth=%s,leaf=%d' % (d, leaf), (lambda d=d, leaf=leaf:
                RandomForestClassifier(n_estimators=500, max_depth=d,
                                       min_samples_leaf=leaf, n_jobs=1,
                                       random_state=RANDOM_STATE))))
    return out


def grid_xgb():
    out = []
    for d in (3, 6):
        for lr in (0.05, 0.2):
            out.append(('depth=%d,lr=%g' % (d, lr), (lambda d=d, lr=lr:
                XGBClassifier(n_estimators=300, max_depth=d, learning_rate=lr,
                              tree_method='hist', n_jobs=1, verbosity=0,
                              eval_metric='logloss',
                              random_state=RANDOM_STATE))))
    return out


def grid_mlp():
    out = []
    for h in ((64,), (128, 64)):
        for a in (1e-4, 1e-2):
            out.append(('h=%s,a=%g' % (h, a), (lambda h=h, a=a: MLPClassifier(
                hidden_layer_sizes=h, alpha=a, early_stopping=True,
                max_iter=300, random_state=RANDOM_STATE))))
    return out


GRIDS = {'logit': grid_logit, 'svm': grid_svm, 'rf': grid_rf, 'mlp': grid_mlp}
if HAVE_XGB:
    GRIDS['xgb'] = grid_xgb
ORDER = ['logit', 'svm', 'rf', 'xgb', 'mlp']


# ------------------------------------------------------------------------ helpers
def first_split(y, g, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(iter(sgkf.split(np.zeros(len(y)), y, g)))


def weight_check(X, y):
    """Step 0. Balanced sample weights must reproduce class_weight='balanced'."""
    print('-' * 78)
    print('STEP 0  weighting equivalence')
    print('-' * 78)
    rng = np.random.default_rng(RANDOM_STATE)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n1, n0 = min(300, len(pos)), min(100, len(neg))
    idx = np.concatenate([rng.choice(pos, n1, replace=False),
                          rng.choice(neg, n0, replace=False)])
    Xs = StandardScaler().fit_transform(X[idx][:, :20])
    ys = y[idx]
    w = compute_sample_weight('balanced', ys)
    print('  subsample  n=%d  class1=%d class0=%d  weight ratio %.2f'
          % (len(ys), n1, n0, w.max() / w.min()))

    a = LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                           random_state=RANDOM_STATE,
                           class_weight='balanced').fit(Xs, ys)
    b = LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                           random_state=RANDOM_STATE).fit(Xs, ys, sample_weight=w)
    dev = float(np.abs(a.coef_ - b.coef_).max())
    scale = float(np.abs(a.coef_).max())
    print('  logistic  max |coef diff| = %.3e   (coef scale %.3e)' % (dev, scale))
    print('  verdict   %s' % ('PASS' if dev < 1e-6 * max(scale, 1.0) else 'FAIL'))

    try:
        p = Pipeline([('sc', StandardScaler()),
                      ('clf', MLPClassifier(hidden_layer_sizes=(8,), max_iter=30,
                                            random_state=RANDOM_STATE))])
        p.fit(Xs, ys, clf__sample_weight=w)
        print('  MLP       sample_weight through Pipeline: PASS')
    except Exception as exc:                                  # noqa: BLE001
        print('  MLP       sample_weight through Pipeline: FAIL -- %s: %s'
              % (type(exc).__name__, exc))
    print('')


def probe(sample, cond, models, seed):
    samples = build_samples()
    if sample not in samples:
        print('unknown sample: %s' % sample)
        return
    loaded = load_condition(cond, samples[sample])
    if loaded is None:
        print('no rows for %s x %s' % (sample, cond))
        return
    X, y, g, n_pool = loaded

    print('=' * 78)
    print('CELL  %s x %s' % (sample, cond))
    print('=' * 78)
    print('  epochs %d   subjects %d   pool %d (+1 length covariate)'
          % (len(y), len(set(g.tolist())), n_pool))

    tr, _ = first_split(y, g, K, seed)
    itr, iva = first_split(y[tr], g[tr], K_INNER, seed)
    Atr, Aval = tr[itr], tr[iva]
    print('  outer train %d   inner train %d   inner val %d\n'
          % (len(tr), len(Atr), len(Aval)))

    t0 = time.time()
    sc = StandardScaler().fit(X[Atr])
    Ztr, Zval = sc.transform(X[Atr]), sc.transform(X[Aval])
    t_scale = time.time() - t0

    t0 = time.time()
    sel = MRMR(k=K_MAX, n_pool=n_pool).fit(Ztr, y[Atr])
    t_mrmr = time.time() - t0
    support = list(sel.support_)
    print('  scaler %.2fs   mRMR(k=%d) %.2fs' % (t_scale, K_MAX, t_mrmr))

    views = {}
    for k in K_GRID:
        cols = list(range(n_pool)) if k is None else support[:k]
        cols = cols + [n_pool]
        views['all' if k is None else str(k)] = (Ztr[:, cols], Zval[:, cols])

    w = compute_sample_weight('balanced', y[Atr])
    rows, totals = [], {}
    for name in ORDER:
        if name not in models:
            continue
        if name not in GRIDS:
            print('  %-5s skipped (xgboost not importable)' % name)
            continue
        sub = 0.0
        for label, factory in GRIDS[name]():
            for kname, (A, B) in views.items():
                t0 = time.time()
                clf = factory()
                clf.fit(A, y[Atr], sample_weight=w)
                clf.predict_proba(B)
                dt = time.time() - t0
                sub += dt
                rows.append((name, label, kname, A.shape[1], dt))
        totals[name] = sub
        print('  %-5s %2d grid x %d k = %3d fits   %8.1fs'
              % (name, len(GRIDS[name]()), len(views),
                 len(GRIDS[name]()) * len(views), sub))
        sys.stdout.flush()

    print('\n  slowest 12 single fits')
    for r in sorted(rows, key=lambda r: -r[4])[:12]:
        print('    %-5s %-16s k=%-3s ncol=%3d  %7.2fs' % r)

    block = t_mrmr + sum(totals.values())
    print('\n' + '-' * 78)
    print('  block (1 cell, 1 seed, 1 outer fold, 1 inner fold) = %.1fs' % block)
    print('  per cell per seed  ~= block x %d x %d x 1.02 = %.1f min'
          % (K, K_INNER, block * K * K_INNER * 1.02 / 60.0))
    print('  models timed: %s' % ', '.join(n for n in ORDER if n in totals))
    print('  single-threaded; the full run parallelises across folds and grid.')
    print('-' * 78 + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='tier2_86')
    ap.add_argument('--cond', default='iii_noica')
    ap.add_argument('--models', nargs='+', default=ORDER)
    ap.add_argument('--seed', type=int, default=RANDOM_STATE)
    ap.add_argument('--skip-check', action='store_true')
    args = ap.parse_args()

    print('=' * 78)
    print('STAGE 3 TIMING PROBE -- no score is computed or printed')
    print('=' * 78 + '\n')

    if not args.skip_check:
        smp = build_samples()
        chk = load_condition('i_brain', smp['tier2_86'])
        if chk is not None:
            weight_check(chk[0], chk[1])

    probe(args.sample, args.cond, set(args.models), args.seed)