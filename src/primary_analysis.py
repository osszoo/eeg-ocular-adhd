"""Stages 8 + 10: unblinded primary analysis with bootstrap confidence intervals.

Run only AFTER the feature set is frozen (git tag features-v1). PREREG 3.7 blinding
ends at that tag; this is the first script that computes ocular arms.

PRE-SPECIFIED PRIMARY CONTRAST (PREREG 3.1)
  dAUC = AUC(neural_ext + length + ocular) - AUC(neural_ext + length)
  84 subjects, Logistic (C=1.0, class_weight balanced), mRMR k=30 inside folds,
  StratifiedGroupKFold K=5, 50 seeds, subject-level aggregation.
  Uncertainty: subject-level PAIRED bootstrap, 2000 resamples, percentile 95% CI.

SECONDARY (PREREG 4): unadjusted contrast, no length in either arm.

STAGE 8 -- DIMENSION PENALTY
  Adding 3 ocular columns also adds 3 dimensions, which by itself moves AUC
  (usually down). A noise arm with 3 standard-normal columns measures that penalty:
      penalty            = AUC(neural+L+noise) - AUC(neural+L)
      penalty-adjusted   = dAUC(ocular) - penalty
  If the penalty is negative, the raw dAUC UNDERSTATES the ocular contribution.

VERDICT RULE (PREREG 3.5, fixed before seeing results)
  CI lower > 0   -> ocular contributes after adjusting for length
  CI contains 0  -> honest null; 84 subjects cannot resolve an effect this size
  CI upper < 0   -> ocular addition is harmful
  If adjusted and unadjusted contrasts disagree -> report as INCONCLUSIVE,
  never pick the favourable one.

AUC here is pooled over subjects using seed-averaged out-of-fold probabilities
(each subject is in exactly one test fold per seed), which is the quantity the
bootstrap resamples. Fold-mean AUC is reported alongside for comparability with
earlier stages.

Note: this CI reflects subject sampling variability, not model-training
variability (PREREG 3.3).

USAGE
  python src/primary_analysis.py                 # 50 seeds, 2000 bootstrap
  python src/primary_analysis.py --seeds 5 --boot 200    # quick check
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                    # noqa: E402
from classify_ext import MRMR, K_SELECT, load_csv       # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
N_BOOT = 2000
OCULAR_COLS = ['ocular_power_ratio', 'ocular_cv', 'ocular_succ_diff']


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


def make_pipe(n_pool):
    """Scaler -> mRMR on the first n_pool columns -> Logistic.

    Columns after n_pool (length, ocular, noise) pass through unselected:
    they are pre-specified covariates, not selection candidates.
    """
    return lambda: Pipeline([
        ('sc', StandardScaler()),
        ('sel', MRMR(k=K_SELECT, n_pool=n_pool)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                   random_state=RANDOM_STATE,
                                   class_weight='balanced'))])


def load_ocular(path):
    """features_ocular.csv -> {subject: (label, array(n_ep, 3))} in epoch order."""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        idx = [header.index(c) for c in OCULAR_COLS]
        out = {}
        for row in r:
            s = row[0]
            if s not in out:
                out[s] = [int(row[1]), []]
            out[s][1].append([float(row[i]) for i in idx])
    return {s: (lab, np.array(v, float)) for s, (lab, v) in out.items()}


def align(Xe, ye, ge, cols, ocular):
    """Restrict the enriched table to subjects that also have an ocular IC.

    Requires identical epoch counts and labels; mismatches are dropped and listed.
    Returns X_neural(306), length(n,1), ocular(n,3), y, groups, dropped.
    """
    i_len = cols.index('rec_length_sec')
    ext_idx = [i for i in range(len(cols)) if i != i_len]
    subs = list(dict.fromkeys(ge.tolist()))
    keep_rows, oc_rows, dropped = [], [], []
    for s in subs:
        rows = np.where(ge == s)[0]
        if s not in ocular:
            continue
        lab, arr = ocular[s]
        if arr.shape[0] != len(rows) or lab != int(ye[rows[0]]):
            dropped.append(s)
            continue
        keep_rows.append(rows)
        oc_rows.append(arr)
    if not keep_rows:
        raise ValueError('no subject aligned between neural and ocular tables')
    rows = np.concatenate(keep_rows)
    OC = np.vstack(oc_rows)
    return (Xe[np.ix_(rows, ext_idx)], Xe[rows][:, [i_len]], OC,
            ye[rows], ge[rows], dropped)


def oof_subject_probs(X, y, groups, make, seeds):
    """Seed-averaged out-of-fold probability per subject.

    Each subject sits in exactly one test fold per seed, so every subject gets one
    probability per seed; these are averaged.
    """
    subs = list(dict.fromkeys(groups.tolist()))
    pos = {s: i for i, s in enumerate(subs)}
    y_sub = np.array([int(y[groups == s][0]) for s in subs])
    total = np.zeros(len(subs))
    count = np.zeros(len(subs))
    fold_aucs = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
            clf = make()
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            gte = groups[te]
            st, sp = [], []
            for s in dict.fromkeys(gte.tolist()):
                m = gte == s
                total[pos[s]] += p[m].mean()
                count[pos[s]] += 1
                st.append(int(y[te][m][0]))
                sp.append(p[m].mean())
            if len(set(st)) == 2:
                fold_aucs.append(roc_auc_score(st, sp))
    return y_sub, total / np.maximum(count, 1), float(np.mean(fold_aucs))


def paired_bootstrap(y_sub, p_a, p_b, n_boot, seed=RANDOM_STATE):
    """Percentile CI for AUC(p_b) - AUC(p_a), resampling SUBJECTS with replacement.

    Paired: both arms are evaluated on the same resampled subjects, which removes
    the shared sampling variance and is what makes the interval informative.
    """
    rng = np.random.default_rng(seed)
    n = len(y_sub)
    d = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        if len(set(y_sub[ii].tolist())) < 2:
            continue
        d.append(roc_auc_score(y_sub[ii], p_b[ii]) - roc_auc_score(y_sub[ii], p_a[ii]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), len(d)


if __name__ == '__main__':
    n_seeds = 50
    n_boot = N_BOOT
    if '--seeds' in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index('--seeds') + 1])
    if '--boot' in sys.argv:
        n_boot = int(sys.argv[sys.argv.index('--boot') + 1])
    seeds = [RANDOM_STATE + i for i in range(n_seeds)]

    out_dir = _RESULTS / 'expansion'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'primary_analysis.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    Xe, ye, ge, cols = load_csv(_RESULTS / 'features_neural_ext.csv')
    ocular = load_ocular(_RESULTS / 'features_ocular.csv')
    NEU, LEN, OC, y, g, dropped = align(Xe, ye, ge, cols, ocular)

    rng = np.random.default_rng(RANDOM_STATE)
    NOISE = rng.standard_normal((NEU.shape[0], 3))
    n_pool = NEU.shape[1]

    print('=== stages 8 + 10: primary analysis (UNBLINDED) ===')
    print('PREREG 3.1 primary: (neural_ext + length + ocular) - (neural_ext + length)')
    print(f'  subjects={len(set(g.tolist()))}  epochs={len(y)}  neural_features={n_pool}')
    print(f'  k={K_SELECT}  K={K}  seeds={n_seeds}  bootstrap={n_boot}')
    if dropped:
        print(f'  dropped for mismatch: {len(dropped)} -> {dropped[:5]}')
    print()

    arms = {
        'neural+L':          np.hstack([NEU, LEN]),
        'neural+L+ocular':   np.hstack([NEU, LEN, OC]),
        'neural+L+noise':    np.hstack([NEU, LEN, NOISE]),
        'neural':            NEU,
        'neural+ocular':     np.hstack([NEU, OC]),
    }

    res = {}
    print(f'  {"arm":<18}{"AUC(pooled)":>13}{"AUC(fold)":>11}{"acc":>8}{"cols":>7}{"sec":>8}')
    for name, X in arms.items():
        t = time.time()
        y_sub, p_sub, fold_auc = oof_subject_probs(X, y, g, make_pipe(n_pool), seeds)
        auc = roc_auc_score(y_sub, p_sub)
        acc = float(((p_sub >= 0.5).astype(int) == y_sub).mean())
        res[name] = (y_sub, p_sub, auc, acc)
        print(f'  {name:<18}{auc:>13.3f}{fold_auc:>11.3f}{acc:>8.3f}'
              f'{X.shape[1]:>7}{time.time() - t:>8.1f}')

    y_sub = res['neural+L'][0]

    def contrast(base, comb, label, star=''):
        m, lo, hi, nb = paired_bootstrap(y_sub, res[base][1], res[comb][1], n_boot)
        sig = 'CI>0' if lo > 0 else ('CI<0' if hi < 0 else 'CI includes 0')
        print(f'  {label:<34}{m:>+8.3f}  [{lo:>+.3f}, {hi:>+.3f}]  {sig}{star}')
        return m, lo, hi

    print('\n[contrasts]  dAUC with 95% paired bootstrap CI')
    prim = contrast('neural+L', 'neural+L+ocular', 'PRIMARY  ocular | length adj.', '  *')
    sec = contrast('neural', 'neural+ocular', 'SECONDARY  ocular | unadjusted')
    pen = contrast('neural+L', 'neural+L+noise', 'stage 8   noise (dimension penalty)')

    print(f'\n[stage 8] penalty-adjusted primary = {prim[0] - pen[0]:+.3f} '
          f'(raw {prim[0]:+.3f} minus penalty {pen[0]:+.3f})')
    if pen[0] < 0:
        print('  penalty is negative -> the raw primary UNDERSTATES the ocular effect.')

    print('\n[PRIMARY VERDICT]  PREREG 3.5')
    if prim[1] > 0:
        print('  CI lower > 0 -> ocular contributes after adjusting for recording length.')
    elif prim[2] < 0:
        print('  CI upper < 0 -> adding ocular features is harmful.')
    else:
        print('  CI includes 0 -> honest null. At n=84 an effect of this size cannot be')
        print('  resolved; report the interval, not a point claim.')
    agree = (prim[1] > 0) == (sec[1] > 0) and (prim[2] < 0) == (sec[2] < 0)
    if not agree:
        print('  ! adjusted and unadjusted contrasts DISAGREE -> report as INCONCLUSIVE')
        print('    (PREREG 3.5 forbids selecting the favourable one).')
    print('\n  * CI covers subject sampling only, not model-training variability (PREREG 3.3).')
    print('  * stopwatch baseline (stage 4): AUC 0.672 / 0.678.')

    sys.stdout = orig
    fh.close()