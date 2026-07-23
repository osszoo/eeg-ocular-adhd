"""Stage 7b: classification on the enriched feature set, with fold-internal mRMR.

Governed by PREREG.md:
  3.6  feature set fixed a priori (306 features); dimension reduction by mRMR with
       k=30 FIXED (no tuning, hence no nested CV); selection strictly INSIDE folds.
  3.7  BLINDING -- while the feature set is being finalised, only neural(+length)
       arms may be inspected. This script therefore contains NO ocular arm.

WHY SELECTION MUST BE FOLD-INTERNAL
  Selecting features on the full dataset leaks the test fold into the choice.
  An audit showed that global selection on pure noise (true AUC 0.500) yields
  AUC 0.741-0.768 -- squarely inside the "honest subject-wise" 78-88% band
  reported in the literature. Fold-internal selection removes that inflation.

mRMR (FCQ variant, the one used by standard implementations)
  relevance  : ANOVA F statistic of each feature against the label
  redundancy : mean |Pearson r| with the already-selected features
  score      : relevance / redundancy, greedy forward selection to k features
  Fitted on TRAINING data only, as a step inside the sklearn Pipeline.

ARMS (blinded)
  neural_base   25 baseline features (region-averaged CLR + TBR)  [if available]
  neural_ext    306 enriched features -> mRMR k=30
  neural_ext+L  306 enriched + recording length -> mRMR k=30 (length always kept)

The +L arm exists because PREREG 3.1 makes recording length a covariate of BOTH
primary arms; here we check what the enriched features add on top of the stopwatch.

USAGE
  python src/classify_ext.py               # 50 seeds (pre-registered)
  python src/classify_ext.py --seeds 5     # quick check
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import evaluate, K, RANDOM_STATE          # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
K_SELECT = 30                      # PREREG 3.6: fixed, not tuned
N_SEEDS_DEFAULT = 50               # PREREG 3.1


class MRMR(BaseEstimator, TransformerMixin):
    """mRMR (FCQ) selector on the first n_pool columns; later columns pass through.

    Pass-through exists because PREREG 3.1 makes recording length a pre-specified
    covariate, not a selection candidate. Feeding it into the selector would let
    mRMR's redundancy penalty discard genuinely informative neural features merely
    for correlating with length -- the opposite of the intended adjustment.
    """

    def __init__(self, k=K_SELECT, n_pool=None):
        self.k = k
        self.n_pool = n_pool

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.n_pool_ = X.shape[1] if self.n_pool is None else int(self.n_pool)
        Xp = X[:, :self.n_pool_]
        pool = list(range(self.n_pool_))

        F, _ = f_classif(Xp, y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        # full |correlation| matrix once (vectorised); constant columns -> 0
        Xc = Xp - Xp.mean(axis=0)
        sd = Xc.std(axis=0)
        sd[sd == 0] = 1.0
        Z = Xc / sd
        R = np.abs(Z.T @ Z) / max(len(X), 1)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(R, 0.0)

        sel = []
        budget = max(self.k, 0)
        if budget and pool:
            first = max(pool, key=lambda i: F[i])
            sel.append(first)
            pool.remove(first)
            budget -= 1
            while budget and pool:
                pool_arr = np.array(pool)
                red = R[np.ix_(pool_arr, np.array(sel))].mean(axis=1)
                red = np.maximum(red, 1e-6)
                best = int(pool_arr[np.argmax(F[pool_arr] / red)])
                sel.append(best)
                pool.remove(best)
                budget -= 1
        self.support_ = np.array(sorted(sel), dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        keep = X[:, self.support_]
        rest = X[:, self.n_pool_:]
        return np.hstack([keep, rest]) if rest.shape[1] else keep


def make_pipe(n_pool=None):
    return lambda: Pipeline([
        ('sc', StandardScaler()),
        ('sel', MRMR(k=K_SELECT, n_pool=n_pool)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                   random_state=RANDOM_STATE,
                                   class_weight='balanced'))])


def load_csv(path):
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    groups = np.array([row[0] for row in rows])
    y = np.array([int(row[1]) for row in rows])
    X = np.array([[float(v) for v in row[2:]] for row in rows])
    return X, y, groups, header[2:]


class _Tee:
    encoding = 'utf-8'

    def __init__(self, *s):
        self.streams = s

    def write(self, x):
        for st in self.streams:
            try:
                st.write(x)
            except (ValueError, OSError):
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except (ValueError, OSError):
                pass


def repeated(X, y, groups, make, seeds):
    """Repeated CV -> per-seed mean AUC and mean accuracy."""
    aucs, accs = [], []
    for s in seeds:
        sp = list(StratifiedGroupKFold(n_splits=K, shuffle=True,
                                       random_state=s).split(np.zeros(len(y)), y, groups))
        m = evaluate(X, y, groups, sp, make)
        aucs.append(np.nanmean(m['auc']))
        accs.append(np.nanmean(m['acc']))
    return np.array(aucs), np.array(accs)


if __name__ == '__main__':
    n_seeds = N_SEEDS_DEFAULT
    if '--seeds' in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index('--seeds') + 1])
    seeds = [RANDOM_STATE + i for i in range(n_seeds)]

    out_dir = _RESULTS / 'expansion'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'classify_ext_blinded.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    Xe, ye, ge, cols = load_csv(_RESULTS / 'features_neural_ext.csv')
    i_len = cols.index('rec_length_sec')
    ext_idx = [i for i in range(len(cols)) if i != i_len]

    print('=== stage 7b: enriched features, fold-internal mRMR (BLINDED) ===')
    print('PREREG 3.7: ocular arms are NOT computed here.')
    print(f'  epochs={len(ye)}  subjects={len(set(ge.tolist()))}  '
          f'features={len(ext_idx)}  k={K_SELECT}  seeds={n_seeds}\n')

    arms = {
        'length only':   ([i_len], make_pipe()),
        'neural_ext':    (ext_idx, make_pipe()),
        'neural_ext+L':  (ext_idx + [i_len], make_pipe(n_pool=len(ext_idx))),
    }

    base = _RESULTS / 'X_neural121.csv'
    if base.exists():
        Xb, yb, gb, _ = load_csv(base)
    else:
        Xb = None

    print(f'  {"arm":<16}{"AUC mean":>10}{"AUC sd":>9}{"acc mean":>10}{"n_feat":>8}{"sec":>8}')
    results = {}
    if Xb is not None:
        t = time.time()
        a, c = repeated(Xb, yb, gb, make_pipe(), seeds)
        results['neural_base'] = (a, c)
        print(f'  {"neural_base":<16}{a.mean():>10.3f}{a.std():>9.3f}{c.mean():>10.3f}'
              f'{Xb.shape[1]:>8}{time.time() - t:>8.1f}')
    for name, (idx, mk) in arms.items():
        t = time.time()
        Xa = Xe[:, idx]
        a, c = repeated(Xa, ye, ge, mk, seeds)
        results[name] = (a, c)
        print(f'  {name:<16}{a.mean():>10.3f}{a.std():>9.3f}{c.mean():>10.3f}'
              f'{Xa.shape[1]:>8}{time.time() - t:>8.1f}')

    print('\n[gain over baseline]  (paired over seeds)')
    if 'neural_base' in results:
        b = results['neural_base'][0]
        for name in ('neural_ext', 'neural_ext+L'):
            d = results[name][0] - b
            print(f'  {name:<16}dAUC {d.mean():>+7.3f}   positive seeds {int((d > 0).sum())}/{n_seeds}')
    print('\n  * stopwatch baseline (stage 4, subject-level): AUC 0.672/0.678')
    print('  * ocular arms remain blinded until the feature set is frozen (features-v1).')

    sys.stdout = orig
    fh.close()