"""Stage 9: confirmatory arms.

Runs after the pre-specified primary (stages 8+10). These are CONFIRMATORY /
EXPLORATORY per PREREG section 4 -- they do not replace the primary verdict, they
probe whether that verdict survives two alternative framings.

ARM 1 -- IS THE OCULAR BLOCK ACTUALLY OCULAR?
  Stage 5a showed the apparent ocular advantage lives almost entirely in cv and
  succ_diff, i.e. in ACROSS-EPOCH VARIABILITY rather than in ocular amplitude.
  If variability per se is what carries the signal, then the same variability
  computed on NEURAL features should reproduce the effect, and nothing about the
  ocular independent component is special.

  Construction: for every one of the 306 enriched features, the within-subject
  standard deviation across that subject's epochs, broadcast back onto the
  subject's rows (exactly how ocular cv/succ_diff are stored). These 306 "neural
  variability" columns join the mRMR candidate pool, so the selector is free to
  prefer them over the ocular block.

  Contrasts:
    (a) neural+L+neuvar        vs neural+L               does neural variability add?
    (b) neural+L+neuvar+ocular vs neural+L+neuvar        does ocular add BEYOND it?
  (b) is the sharpest test in the whole study: if it collapses while (a) is large,
  the ocular claim reduces to "any across-epoch variability will do".
  SD is used rather than CV because CLR features can have near-zero means, which
  makes a ratio unstable; SD is the same notion of variability without that flaw.

ARM 2 -- DID EPOCH-LEVEL AGGREGATION CREATE THE RESULT?
  The main pipeline trains on epochs and averages probabilities per subject.
  Here each subject becomes ONE row (mean of each feature over that subject's
  epochs), giving an 84 x N table classified directly, with the same model and
  selection rule. This matches the subject-level table structure used by
  Garcia-Ponsoda et al. and removes epoch-count imbalance by construction.
  Because rows are subjects, plain StratifiedKFold is used (no grouping needed).

Uncertainty: subject-level paired bootstrap, as in the primary analysis.

USAGE
  python src/confirm_arms.py                     # 50 seeds, 2000 bootstrap
  python src/confirm_arms.py --seeds 5 --boot 200
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                                  # noqa: E402
from classify_ext import K_SELECT, load_csv                           # noqa: E402
from primary_analysis import (make_pipe, load_ocular, align,          # noqa: E402
                              oof_subject_probs, paired_bootstrap, _Tee)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'


def within_subject_sd(X, groups):
    """Per-subject SD of each column across that subject's epochs, broadcast back.

    Mirrors how ocular cv / succ_diff are stored: one value per subject, repeated
    on every row belonging to that subject.
    """
    out = np.zeros_like(X, dtype=float)
    for s in dict.fromkeys(groups.tolist()):
        m = groups == s
        out[m] = X[m].std(axis=0, ddof=1) if m.sum() > 1 else 0.0
    return out


def subject_table(X, y, groups):
    """Collapse epoch rows to one row per subject (column means)."""
    subs = list(dict.fromkeys(groups.tolist()))
    Xs = np.vstack([X[groups == s].mean(axis=0) for s in subs])
    ys = np.array([int(y[groups == s][0]) for s in subs])
    return Xs, ys, subs


def oof_subject_level(Xs, ys, make, seeds):
    """Repeated stratified CV on the 84-row table -> seed-averaged OOF probability.

    Rows are subjects, so no grouping is required; each row is held out once per seed.
    """
    total = np.zeros(len(ys))
    count = np.zeros(len(ys))
    fold_aucs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xs, ys):
            clf = make()
            clf.fit(Xs[tr], ys[tr])
            p = clf.predict_proba(Xs[te])[:, 1]
            total[te] += p
            count[te] += 1
            if len(set(ys[te].tolist())) == 2:
                fold_aucs.append(roc_auc_score(ys[te], p))
    return total / np.maximum(count, 1), float(np.mean(fold_aucs))


if __name__ == '__main__':
    n_seeds, n_boot = 50, 2000
    if '--seeds' in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index('--seeds') + 1])
    if '--boot' in sys.argv:
        n_boot = int(sys.argv[sys.argv.index('--boot') + 1])
    seeds = [RANDOM_STATE + i for i in range(n_seeds)]

    out_dir = _RESULTS / 'expansion'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'confirm_arms.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    Xe, ye, ge, cols = load_csv(_RESULTS / 'features_neural_ext.csv')
    ocular = load_ocular(_RESULTS / 'features_ocular.csv')
    NEU, LEN, OC, y, g, dropped = align(Xe, ye, ge, cols, ocular)
    NEUVAR = within_subject_sd(NEU, g)
    n_neu = NEU.shape[1]

    print('=== stage 9: confirmatory arms ===')
    print(f'  subjects={len(set(g.tolist()))}  epochs={len(y)}  '
          f'neural={n_neu}  neuvar={NEUVAR.shape[1]}  k={K_SELECT}  '
          f'seeds={n_seeds}  bootstrap={n_boot}')
    if dropped:
        print(f'  dropped for mismatch: {len(dropped)}')
    print()

    # ---------------- ARM 1: neural variability control (epoch level)
    arms1 = {
        'neural+L':                 (np.hstack([NEU, LEN]), n_neu),
        'neural+L+neuvar':          (np.hstack([NEU, NEUVAR, LEN]), n_neu * 2),
        'neural+L+neuvar+ocular':   (np.hstack([NEU, NEUVAR, LEN, OC]), n_neu * 2),
        'neural+L+ocular':          (np.hstack([NEU, LEN, OC]), n_neu),
    }
    print('[arm 1] neural-variability control  (epoch level, mRMR pool includes neuvar)')
    print(f'  {"arm":<26}{"AUC(pooled)":>13}{"AUC(fold)":>11}{"acc":>8}{"cols":>7}{"sec":>8}')
    r1 = {}
    for name, (X, pool) in arms1.items():
        t = time.time()
        y_sub, p_sub, fauc = oof_subject_probs(X, y, g, make_pipe(pool), seeds)
        auc = roc_auc_score(y_sub, p_sub)
        acc = float(((p_sub >= 0.5).astype(int) == y_sub).mean())
        r1[name] = (y_sub, p_sub, auc, acc)
        print(f'  {name:<26}{auc:>13.3f}{fauc:>11.3f}{acc:>8.3f}{X.shape[1]:>7}'
              f'{time.time() - t:>8.1f}')

    y_sub = r1['neural+L'][0]

    def contrast(base, comb, label, store, note=''):
        m, lo, hi, _ = paired_bootstrap(y_sub, store[base][1], store[comb][1], n_boot)
        sig = 'CI>0' if lo > 0 else ('CI<0' if hi < 0 else 'CI includes 0')
        print(f'  {label:<40}{m:>+8.3f}  [{lo:>+.3f}, {hi:>+.3f}]  {sig}{note}')
        return m, lo, hi

    print('\n  dAUC with 95% paired bootstrap CI')
    a = contrast('neural+L', 'neural+L+neuvar',
                 '(a) neural variability | length adj.', r1)
    b = contrast('neural+L+neuvar', 'neural+L+neuvar+ocular',
                 '(b) ocular BEYOND neural variability', r1, '  <-')
    c = contrast('neural+L', 'neural+L+ocular',
                 '    ocular | length adj. (primary, repeat)', r1)

    print('\n  [reading] if (a) is sizeable and (b) collapses, the ocular block is')
    print('  interchangeable with generic across-epoch variability, and its specificity')
    print('  to the ocular component is not supported.')

    # ---------------- ARM 2: subject-level table (84 rows)
    print('\n[arm 2] subject-level table (one row per subject, StratifiedKFold)')
    Xn_s, ys, subs = subject_table(NEU, y, g)
    Xl_s, _, _ = subject_table(LEN, y, g)
    Xo_s, _, _ = subject_table(OC, y, g)
    arms2 = {
        'neural+L':        (np.hstack([Xn_s, Xl_s]), n_neu),
        'neural+L+ocular': (np.hstack([Xn_s, Xl_s, Xo_s]), n_neu),
    }
    print(f'  rows={Xn_s.shape[0]}  ADHD={int(ys.sum())}  Control={int((1 - ys).sum())}')
    print(f'  {"arm":<26}{"AUC(pooled)":>13}{"AUC(fold)":>11}{"acc":>8}{"cols":>7}{"sec":>8}')
    r2 = {}
    for name, (X, pool) in arms2.items():
        t = time.time()
        p_sub, fauc = oof_subject_level(X, ys, make_pipe(pool), seeds)
        auc = roc_auc_score(ys, p_sub)
        acc = float(((p_sub >= 0.5).astype(int) == ys).mean())
        r2[name] = (ys, p_sub, auc, acc)
        print(f'  {name:<26}{auc:>13.3f}{fauc:>11.3f}{acc:>8.3f}{X.shape[1]:>7}'
              f'{time.time() - t:>8.1f}')

    m, lo, hi, _ = paired_bootstrap(ys, r2['neural+L'][1], r2['neural+L+ocular'][1], n_boot)
    sig = 'CI>0' if lo > 0 else ('CI<0' if hi < 0 else 'CI includes 0')
    print('\n  dAUC with 95% paired bootstrap CI')
    print(f'  {"ocular | length adj. (subject level)":<40}{m:>+8.3f}  '
          f'[{lo:>+.3f}, {hi:>+.3f}]  {sig}')

    # ---------------- summary
    print('\n[summary]')
    print(f'  epoch-level primary (repeat) : {c[0]:+.3f}  [{c[1]:+.3f}, {c[2]:+.3f}]')
    print(f'  subject-level equivalent     : {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
    print(f'  ocular beyond neural variab. : {b[0]:+.3f}  [{b[1]:+.3f}, {b[2]:+.3f}]')
    agree = (c[1] > 0) == (lo > 0) and (c[2] < 0) == (hi < 0)
    print(f'  aggregation methods agree    : {"yes" if agree else "NO -> report both"}')
    print('\n  * confirmatory, not a substitute for the pre-registered primary verdict.')

    sys.stdout = orig
    fh.close()