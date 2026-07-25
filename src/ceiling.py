"""Stage 3 ceiling measurement (PREREG 8.3.4).

Measures how high AUC can go on the fixed 306-feature set, WITHOUT touching the
stage-4 verdict on ocular information (that is closed by 8.3.1). Two axes, never
mixed, crossed as a star design around the stage-4 primary centre point
(Logistic, C=1.0, mRMR k=30, K=5):

  MODEL AXIS  (8.3.4(1a))  outer K=5 fixed; sweep 5 models x hyperparameters,
                          chosen by a hand-written inner loop on SUBJECT-level
                          AUC (GridSearchCV cannot, because its scoring is blind
                          to groups and would select on epoch-level AUC, letting
                          the 0.545 epoch prior back in).
  SAMPLE AXIS (8.3.4(11)) hyperparameters FROZEN at the centre point; sweep only
                          outer K in {5, 10, LOSO}. This is a learning curve, so
                          model and hp must stay fixed or the K effect and a
                          re-optimisation effect become inseparable. No inner CV.

CELLS (8.3.4(2))  6 conditions x defined samples = 28.
  non-ocular  i_brain iii_noica iv_noasr v_wideband   x all six samples   = 24
  ocular      ii_eye1 ii_eyeall                        x roster84,roster59 =  4

LEAK CONTROL (the two ways nested CV silently cheats)
  1. mRMR sees inner-train ONLY. Fitting it on inner-train+val would leak val
     into selection and inflate the hp choice made on that same val.
  2. StandardScaler is refit per fold (inner-train for selection, outer-train for
     the final refit). A scaler fit once outside leaks test statistics into train.

ORDERED SELECTION (8.3.4(4c))  classify_ext.MRMR sorts support_ and loses the
  order prefixes need. MRMROrdered subclasses it, keeps order_, and takes
  sorted(order_[:k]) so k in {15,30,60} all come from ONE k_max=60 fit per
  (inner fold). At k=30 sorted(order_[:30]) must equal the stage-4 support_;
  --verify checks this.

SELF-VERIFICATION (8.3.4(8), run before any real measurement)
  --verify checks three identities, in the spirit of stage-2 ov00 (+-0.000) and
  stage-4's apply(exclude=[]) identity test:
    A  MRMROrdered prefix at k=30 == stage-4 MRMR support_  (element-wise)
    B  centre-point pooled AUC (10 seeds, no tuning) reproduces the stage-4
       i_brain estimator recomputed on the SAME 10 seeds. Not the cached 50-seed
       number: 10 != 50, so the target is stage-4 logic on 10 seeds, not its cache.
    C  balanced sample_weight reproduces class_weight='balanced' (coef diff 0).

OUTPUTS (8.3.4(6),(7))  two CSVs, appended, resumable:
  results/ceiling/scores.csv      one row per (cell, axis, K, seed) -- performance
  results/ceiling/selections.csv  one row per (cell, seed, outer fold) chosen hp
                                  -- the grid-boundary diagnostic. Without it the
                                  ceiling claim does not stand (8.3.4(7)).

NOT DONE HERE  the "max + expected-max-under-null (selection discount)" frame is
  deliberately absent for the 28-cell main report (8.3.4(10)); only the combo arm
  carries a discount, per 8.3.3(4). This file computes real AUCs and nothing else.

USAGE
  C:\\envs\\eeg\\python.exe src/ceiling.py --verify
  C:\\envs\\eeg\\python.exe src/ceiling.py --smoke              # 1 cell, 1 seed, model axis
  C:\\envs\\eeg\\python.exe src/ceiling.py --axis model --seeds 10
  C:\\envs\\eeg\\python.exe src/ceiling.py --axis sample --seeds 10
  C:\\envs\\eeg\\python.exe src/ceiling.py --axis both --seeds 10   # full stage 3
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                              # noqa: E402
from classify_ext import MRMR, K_SELECT, load_csv                 # noqa: E402
from contrast_ic import (build_samples, load_condition,           # noqa: E402
                         _COND_DIR, _Tee)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

warnings.filterwarnings('ignore', category=ConvergenceWarning)

_CEIL_DIR = _ROOT / 'results' / 'ceiling'
_SCORES = _CEIL_DIR / 'scores.csv'
_SELECT = _CEIL_DIR / 'selections.csv'

K_INNER = 3
K_MAX = 60
K_GRID = [15, 30, 60, None]                # None -> no selection (all 306)

# Centre point = stage-4 primary. Sample axis freezes hp here (8.3.4(11)).
CENTRE_C = 1.0
CENTRE_K = K_SELECT                         # 30

NON_OCULAR = ['i_brain', 'iii_noica', 'iv_noasr', 'v_wideband']
OCULAR = ['ii_eye1', 'ii_eyeall']
ALL_SAMPLES = ['full121', 'tier1_91', 'tier2_86', 'tact120', 'roster84', 'roster59']
ROSTER_SAMPLES = ['roster84', 'roster59']

# Stage-4 i_brain pooled AUC (50 seeds) -- reference only, printed beside the
# 10-seed centre point so the seed-count difference is visible. NOT the --verify
# target (see docstring B).
STAGE4_IBRAIN_50 = {'full121': 0.798, 'tier1_91': 0.807, 'tier2_86': 0.785,
                    'tact120': 0.794, 'roster84': 0.713, 'roster59': 0.720}

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


# ----------------------------------------------------- ordered mRMR (8.3.4(4c))
class MRMROrdered(MRMR):
    """MRMR that remembers selection order so prefixes are valid.

    The parent stores support_ = sorted(sel), discarding order. We re-run the
    identical FCQ loop but keep the order, then expose sorted(order_[:k]) via
    prefix(). At k = self.k the prefix must equal the parent's support_ on the
    same data; verified in --verify test A.
    """

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.n_pool_ = X.shape[1] if self.n_pool is None else int(self.n_pool)
        Xp = X[:, :self.n_pool_]
        pool = list(range(self.n_pool_))

        from sklearn.feature_selection import f_classif
        F, _ = f_classif(Xp, y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        Xc = Xp - Xp.mean(axis=0)
        sd = Xc.std(axis=0)
        sd[sd == 0] = 1.0
        Z = Xc / sd
        R = np.abs(Z.T @ Z) / max(len(X), 1)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(R, 0.0)

        order = []
        budget = max(self.k, 0)
        if budget and pool:
            first = max(pool, key=lambda i: F[i])
            order.append(first)
            pool.remove(first)
            budget -= 1
            while budget and pool:
                pa = np.array(pool)
                red = np.maximum(R[np.ix_(pa, np.array(order))].mean(axis=1), 1e-6)
                best = int(pa[np.argmax(F[pa] / red)])
                order.append(best)
                pool.remove(best)
                budget -= 1
        self.order_ = np.array(order, dtype=int)         # selection order kept
        self.support_ = np.array(sorted(order), dtype=int)
        return self

    def prefix(self, k):
        """Column indices for the first k selected features, plus pass-through."""
        take = self.order_ if k is None else self.order_[:k]
        return np.array(sorted(take.tolist()), dtype=int)


# ------------------------------------------------------------- model grids (8.3.4(3))
def grid_logit():
    return [('logit', 'C=%g' % c, dict(C=c)) for c in (0.1, 1.0, 10.0)]


def grid_svm():
    return [('svm', 'C=%g,g=%s' % (c, g), dict(C=c, gamma=g))
            for c in (1.0, 10.0) for g in ('scale', 0.01)]


def grid_rf():
    return [('rf', 'depth=%s,leaf=%d' % (d, lf), dict(max_depth=d, min_samples_leaf=lf))
            for d in (None, 8) for lf in (1, 5)]


def grid_xgb():
    return [('xgb', 'depth=%d,lr=%g' % (d, lr), dict(max_depth=d, learning_rate=lr))
            for d in (3, 6) for lr in (0.05, 0.2)]


def grid_mlp():
    return [('mlp', 'h=%s,a=%g' % (h, a), dict(hidden_layer_sizes=h, alpha=a))
            for h in ((64,), (128, 64)) for a in (1e-4, 1e-2)]


def build_estimator(model, hp):
    """One fresh estimator. class balance via sample_weight at fit time, uniform
    across models (8.3.4(5)); nothing here carries class_weight."""
    if model == 'logit':
        return LogisticRegression(penalty='l2', max_iter=1000,
                                  random_state=RANDOM_STATE, **hp)
    if model == 'svm':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **hp)
    if model == 'rf':
        return RandomForestClassifier(n_estimators=500, n_jobs=1,
                                      random_state=RANDOM_STATE, **hp)
    if model == 'xgb':
        return XGBClassifier(n_estimators=300, tree_method='hist', n_jobs=1,
                             verbosity=0, eval_metric='logloss',
                             random_state=RANDOM_STATE, **hp)
    if model == 'mlp':
        return MLPClassifier(early_stopping=True, max_iter=300,
                             random_state=RANDOM_STATE, **hp)
    raise ValueError(model)


def model_grid(models):
    grids = {'logit': grid_logit, 'svm': grid_svm, 'rf': grid_rf, 'mlp': grid_mlp}
    if HAVE_XGB:
        grids['xgb'] = grid_xgb
    order = ['logit', 'svm', 'rf', 'xgb', 'mlp']
    out = []
    for name in order:
        if name in models and name in grids:
            out += grids[name]()
    return out


# ------------------------------------------------------------------ subject AUC
def subject_auc(y_epoch, p_epoch, groups):
    """Epoch prob -> subject mean -> AUC. The evaluation the whole project uses;
    the inner loop selects on THIS, not epoch AUC (8.3.4(4a))."""
    subs = list(dict.fromkeys(groups.tolist()))
    ys = np.array([int(y_epoch[groups == s][0]) for s in subs])
    ps = np.array([p_epoch[groups == s].mean() for s in subs])
    if len(set(ys.tolist())) < 2:
        return float('nan'), ys, ps, subs
    return float(roc_auc_score(ys, ps)), ys, ps, subs


def proba(est, Xtr, ytr, w, Xte):
    est.fit(Xtr, ytr, sample_weight=w)
    return est.predict_proba(Xte)[:, 1]


# --------------------------------------------------- nested inner loop (model axis)
def inner_select(Xtr, ytr, gtr, n_pool, grid, seed):
    """Pick (model, hp, k) by mean subject AUC over K_INNER inner folds.

    Per inner fold: mRMR k_max=60 on INNER-TRAIN only (leak control 1), scaler on
    inner-train only (leak control 2), then every (grid point x k prefix) scored
    on inner-val at subject level. mRMR is fit once per inner fold and all k
    prefixes reuse it (8.3.4(4c)).
    """
    isgkf = StratifiedGroupKFold(n_splits=K_INNER, shuffle=True, random_state=seed)
    # accumulate subject-AUC per candidate across inner folds
    acc = {}
    for itr, iva in isgkf.split(np.zeros(len(ytr)), ytr, gtr):
        sc = StandardScaler().fit(Xtr[itr])
        Zi, Zv = sc.transform(Xtr[itr]), sc.transform(Xtr[iva])
        sel = MRMROrdered(k=K_MAX, n_pool=n_pool).fit(Zi, ytr[itr])
        wi = compute_sample_weight('balanced', ytr[itr])
        for k in K_GRID:
            cols = list(sel.prefix(k)) + [n_pool]      # + length pass-through
            Ai, Av = Zi[:, cols], Zv[:, cols]
            for model, label, hp in grid:
                est = build_estimator(model, hp)
                p = proba(est, Ai, ytr[itr], wi, Av)
                a, _, _, _ = subject_auc(ytr[iva], p, gtr[iva])
                key = (model, label, 'all' if k is None else str(k))
                acc.setdefault(key, []).append(a)
    best_key, best_mean = None, -1.0
    for key, vals in acc.items():
        m = float(np.nanmean(vals))
        if m > best_mean:
            best_key, best_mean = key, m
    return best_key, best_mean          # (model, label, kname)


def refit_and_predict(Xtr, ytr, gtr, Xte, n_pool, model, label, kname, hp):
    """Refit the winning candidate on the WHOLE outer-train, predict outer-test.
    Scaler and mRMR refit on outer-train (leak control 2)."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    k = None if kname == 'all' else int(kname)
    sel = MRMROrdered(k=K_MAX, n_pool=n_pool).fit(Ztr, ytr)
    cols = list(sel.prefix(k)) + [n_pool]
    w = compute_sample_weight('balanced', ytr)
    est = build_estimator(model, hp)
    return proba(est, Ztr[:, cols], ytr, w, Zte[:, cols])


HP_BY_LABEL = None
def _hp_lookup(model, label):
    global HP_BY_LABEL
    if HP_BY_LABEL is None:
        HP_BY_LABEL = {}
        for m, lb, hp in (grid_logit() + grid_svm() + grid_rf() +
                          (grid_xgb() if HAVE_XGB else []) + grid_mlp()):
            HP_BY_LABEL[(m, lb)] = hp
    return HP_BY_LABEL[(model, label)]


# ------------------------------------------------------------- one cell, model axis
def run_model_axis(sample, cond, keep, seed, grid, models_str):
    loaded = load_condition(cond, keep)
    if loaded is None:
        return None
    X, y, g, n_pool = loaded
    osgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)

    subs = list(dict.fromkeys(g.tolist()))
    tot = np.zeros(len(subs)); cnt = np.zeros(len(subs))
    pos = {s: i for i, s in enumerate(subs)}
    ysub = np.array([int(y[g == s][0]) for s in subs])
    picks = []
    for fold, (tr, te) in enumerate(osgkf.split(np.zeros(len(y)), y, g)):
        key, _ = inner_select(X[tr], y[tr], g[tr], n_pool, grid, seed)
        model, label, kname = key
        hp = _hp_lookup(model, label)
        p = refit_and_predict(X[tr], y[tr], g[tr], X[te], n_pool,
                              model, label, kname, hp)
        gte = g[te]
        for s in dict.fromkeys(gte.tolist()):
            m = gte == s
            tot[pos[s]] += p[m].mean(); cnt[pos[s]] += 1
        picks.append((fold, model, label, kname))
    psub = tot / np.maximum(cnt, 1)
    auc = float(roc_auc_score(ysub, psub)) if len(set(ysub.tolist())) == 2 else float('nan')
    acc, sens, spec = _rates(ysub, psub)
    return dict(auc=auc, acc=acc, sens=sens, spec=spec, n=len(subs), picks=picks)


# --------------------------------------------------------- one cell, sample axis
def run_sample_axis(sample, cond, keep, seed, outer_K):
    """Centre-point hp frozen (8.3.4(11)); only outer K varies. No inner CV."""
    loaded = load_condition(cond, keep)
    if loaded is None:
        return None
    X, y, g, n_pool = loaded
    if outer_K == 'loso':
        splitter = LeaveOneGroupOut().split(np.zeros(len(y)), y, g)
    else:
        sk = StratifiedGroupKFold(n_splits=int(outer_K), shuffle=True, random_state=seed)
        splitter = sk.split(np.zeros(len(y)), y, g)

    subs = list(dict.fromkeys(g.tolist()))
    tot = np.zeros(len(subs)); cnt = np.zeros(len(subs))
    pos = {s: i for i, s in enumerate(subs)}
    ysub = np.array([int(y[g == s][0]) for s in subs])
    for tr, te in splitter:
        p = refit_and_predict(X[tr], y[tr], g[tr], X[te], n_pool,
                              'logit', 'C=%g' % CENTRE_C, str(CENTRE_K),
                              dict(C=CENTRE_C))
        gte = g[te]
        for s in dict.fromkeys(gte.tolist()):
            m = gte == s
            tot[pos[s]] += p[m].mean(); cnt[pos[s]] += 1
    psub = tot / np.maximum(cnt, 1)
    auc = float(roc_auc_score(ysub, psub)) if len(set(ysub.tolist())) == 2 else float('nan')
    acc, sens, spec = _rates(ysub, psub)
    return dict(auc=auc, acc=acc, sens=sens, spec=spec, n=len(subs))


def _rates(ysub, psub):
    pred = (np.asarray(psub) >= 0.5).astype(int)
    y = np.asarray(ysub).astype(int)
    acc = float((pred == y).mean())
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float('nan')
    spec = float((1 - pred[y == 0]).mean()) if (y == 0).any() else float('nan')
    return acc, sens, spec


# --------------------------------------------------------------------- csv append
def _ensure(path, header):
    _CEIL_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(header)


def _done_keys(path, key_cols):
    if not path.exists():
        return set()
    out = set()
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            out.add(tuple(row[c] for c in key_cols))
    return out


SCORE_HEADER = ['sample', 'cond', 'axis', 'outer_K', 'seed', 'n_sub',
                'auc', 'acc', 'sens', 'spec', 'best_model', 'best_hp',
                'best_k', 'elapsed_s']
SEL_HEADER = ['sample', 'cond', 'seed', 'outer_fold', 'model', 'hp', 'k']


# ---------------------------------------------------------------- verification
def verify(seeds):
    print('=' * 78)
    print('SELF-VERIFICATION (PREREG 8.3.4(8)) -- must pass before measurement')
    print('=' * 78)
    samples = build_samples()
    keep = samples['tier2_86']
    loaded = load_condition('i_brain', keep)
    if loaded is None:
        print('  cannot load i_brain x tier2_86; aborting'); return False
    X, y, g, n_pool = loaded
    ok = True

    # A -- ordered prefix at k=30 equals stage-4 MRMR support
    print('\n[A] MRMROrdered prefix(k=30) == stage-4 MRMR support_')
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    base = MRMR(k=CENTRE_K, n_pool=n_pool).fit(Z, y)
    ordd = MRMROrdered(k=K_MAX, n_pool=n_pool).fit(Z, y)
    pref = ordd.prefix(CENTRE_K)
    same = np.array_equal(np.sort(base.support_), np.sort(pref))
    print('    stage-4 support (%d cols): %s' % (len(base.support_),
          np.sort(base.support_)[:8]))
    print('    ordered prefix  (%d cols): %s' % (len(pref), np.sort(pref)[:8]))
    print('    match: %s' % ('PASS' if same else 'FAIL'))
    ok = ok and same

    # C -- balanced sample_weight == class_weight='balanced' on logistic
    print('\n[C] balanced sample_weight == class_weight=balanced (logistic)')
    w = compute_sample_weight('balanced', y)
    a = LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                           random_state=RANDOM_STATE,
                           class_weight='balanced').fit(Z[:, :20], y)
    b = LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                           random_state=RANDOM_STATE).fit(Z[:, :20], y, sample_weight=w)
    dev = float(np.abs(a.coef_ - b.coef_).max())
    print('    max |coef diff| = %.3e   %s' % (dev, 'PASS' if dev < 1e-9 else 'FAIL'))
    ok = ok and dev < 1e-9

    # B -- centre point (10 seeds, no tuning) reproduces stage-4 logic on 10 seeds
    print('\n[B] centre-point pooled AUC (%d seeds) vs stage-4 i_brain logic' % len(seeds))
    print('    NOTE: stage-4 cache is 50 seeds; target is stage-4 estimator on the')
    print('    SAME %d seeds, recomputed here. The 50-seed cache (0.785) is printed' % len(seeds))
    print('    only for scale.')
    subs = list(dict.fromkeys(g.tolist()))
    pos = {s: i for i, s in enumerate(subs)}
    ysub = np.array([int(y[g == s][0]) for s in subs])
    # centre point via this file's sample-axis path (frozen hp, K=5)
    r = run_sample_axis('tier2_86', 'i_brain', keep, seeds[0], 5)
    # recompute with multi-seed averaging to match stage-4 seed-averaging
    tot = np.zeros(len(subs)); cnt = np.zeros(len(subs))
    for sd in seeds:
        sk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
        for tr, te in sk.split(np.zeros(len(y)), y, g):
            p = refit_and_predict(X[tr], y[tr], g[tr], X[te], n_pool,
                                  'logit', 'C=1', str(CENTRE_K), dict(C=1.0))
            gte = g[te]
            for s in dict.fromkeys(gte.tolist()):
                m = gte == s
                tot[pos[s]] += p[m].mean(); cnt[pos[s]] += 1
    psub = tot / np.maximum(cnt, 1)
    auc10 = float(roc_auc_score(ysub, psub))
    print('    centre point (%d-seed averaged): AUC = %.4f' % (len(seeds), auc10))
    print('    stage-4 cache (50-seed):         AUC = %.4f  [scale ref only]'
          % STAGE4_IBRAIN_50['tier2_86'])
    print('    -> record this number; compare against a stage-4 rerun on the same')
    print('       %d seeds (contrast_ic.py --seeds %d) for the exact identity.'
          % (len(seeds), len(seeds)))

    print('\n' + '-' * 78)
    print('VERIFICATION: %s' % ('ALL AUTOMATED CHECKS PASS' if ok
                                else 'FAILURE ABOVE -- do not run measurement'))
    print('  (B is a recorded value to match against a same-seed stage-4 rerun,')
    print('   not an automated assert, because the cache is 50-seed.)')
    print('-' * 78)
    return ok


# ---------------------------------------------------------------------- drivers
def cells_non_ocular():
    return [(s, c) for c in NON_OCULAR for s in ALL_SAMPLES]


def cells_ocular():
    return [(s, c) for c in OCULAR for s in ROSTER_SAMPLES]


def all_cells():
    return cells_non_ocular() + cells_ocular()


def drive_model_axis(seeds, models, cells, smoke):
    _ensure(_SCORES, SCORE_HEADER)
    _ensure(_SELECT, SEL_HEADER)
    done = _done_keys(_SCORES, ['sample', 'cond', 'axis', 'outer_K', 'seed'])
    samples = build_samples()
    grid = model_grid(models)
    t0 = time.time()
    for seed in seeds:
        for (s, c) in cells:
            key = (s, c, 'model', '5', str(seed))
            if key in done:
                print('[skip] %-10s %-11s seed=%d (cached)' % (s, c, seed)); continue
            r = run_model_axis(s, c, samples[s], seed, grid, models)
            if r is None:
                print('[----] %-10s %-11s seed=%d no rows' % (s, c, seed)); continue
            with open(_SCORES, 'a', newline='', encoding='utf-8') as fh:
                # modal (model,label,k) as ONE tuple, so the summary is a real
                # pick that occurred, not three independent modes glued together
                triples = [(p[1], p[2], p[3]) for p in r['picks']]
                bm, bl, bk = max(set(triples), key=triples.count)
                csv.writer(fh).writerow([s, c, 'model', 5, seed, r['n'],
                    '%.4f' % r['auc'], '%.4f' % r['acc'], '%.4f' % r['sens'],
                    '%.4f' % r['spec'], bm, bl, bk, '%.1f' % (time.time() - t0)])
            with open(_SELECT, 'a', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                for fold, model, label, kname in r['picks']:
                    w.writerow([s, c, seed, fold, model, label, kname])
            print('[ok  ] %-10s %-11s seed=%d AUC=%.4f  %6.1fs'
                  % (s, c, seed, r['auc'], time.time() - t0))
            sys.stdout.flush()
            if smoke:
                print('smoke: stop after first cell'); return


def drive_sample_axis(seeds, cells, smoke):
    _ensure(_SCORES, SCORE_HEADER)
    done = _done_keys(_SCORES, ['sample', 'cond', 'axis', 'outer_K', 'seed'])
    samples = build_samples()
    t0 = time.time()
    for seed in seeds:
        for (s, c) in cells:
            for outer_K in ('5', '10', 'loso'):
                key = (s, c, 'sample', outer_K, str(seed))
                if key in done:
                    print('[skip] %-10s %-11s K=%-4s seed=%d' % (s, c, outer_K, seed))
                    continue
                # LOSO is deterministic; run it once (seed of record = first seed)
                if outer_K == 'loso' and seed != seeds[0]:
                    continue
                r = run_sample_axis(s, c, samples[s], seed, outer_K)
                if r is None:
                    print('[----] %-10s %-11s K=%s no rows' % (s, c, outer_K)); continue
                with open(_SCORES, 'a', newline='', encoding='utf-8') as fh:
                    csv.writer(fh).writerow([s, c, 'sample', outer_K, seed, r['n'],
                        '%.4f' % r['auc'], '%.4f' % r['acc'], '%.4f' % r['sens'],
                        '%.4f' % r['spec'], 'logit', 'C=%g' % CENTRE_C,
                        str(CENTRE_K), '%.1f' % (time.time() - t0)])
                print('[ok  ] %-10s %-11s K=%-4s seed=%d AUC=%.4f  %6.1fs'
                      % (s, c, outer_K, seed, r['auc'], time.time() - t0))
                sys.stdout.flush()
            if smoke:
                print('smoke: stop after first cell'); return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--axis', choices=['model', 'sample', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--models', nargs='+',
                    default=['logit', 'svm', 'rf', 'xgb', 'mlp'])
    ap.add_argument('--ocular-only', action='store_true',
                    help='restrict to the 4 ocular cells')
    ap.add_argument('--non-ocular-only', action='store_true')
    args = ap.parse_args()

    seeds = [RANDOM_STATE + i for i in range(args.seeds)]

    _CEIL_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    logname = 'verify.txt' if args.verify else ('ceiling_%s.txt' % args.axis)
    with open(_CEIL_DIR / logname, 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            if args.verify:
                verify(seeds)
            else:
                if args.ocular_only:
                    cells = cells_ocular()
                elif args.non_ocular_only:
                    cells = cells_non_ocular()
                else:
                    cells = all_cells()
                if args.axis in ('model', 'both'):
                    drive_model_axis(seeds, set(args.models), cells, args.smoke)
                if args.axis in ('sample', 'both'):
                    drive_sample_axis(seeds, cells, args.smoke)
        finally:
            sys.stdout = orig