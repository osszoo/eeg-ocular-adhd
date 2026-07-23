"""Stage 5b confounder gate: epoch equalization -- the decisive test.

Stage 4: recording length alone reaches AUC ~0.68.
Stage 5a: the combined advantage lives almost entirely in cv/succ_diff, not in the
length-robust power_ratio. Consistent with length leaking through the variability
features -- but not proof: cv/succ may carry genuine ocular instability. This decides.

MANIPULATION: truncate every subject to the same N epochs, then RECOMPUTE cv and
succ_diff from the truncated series. Then no feature can encode epoch count
(= recording length), and the epoch-level class prior returns to subject-level.
N = min epochs over the 84 subjects, so NO subject is dropped; a larger N would drop
the shortest subjects, who are concentrated in the control group, re-introducing the
same confound as differential dropout. Small N makes cv/succ noisy for everyone
equally -- noise, not bias. Window: first N (primary), evenly-spaced (sensitivity).

PRE-SPECIFIED PRIMARY CONTRAST (fixed before seeing results):
  dAUC = AUC(neural+P+C+S) - AUC(neural), equalized at N=min, first-N, AUC, 3 clfs.
    survives positive -> ocular contributes with length physically blocked
    collapses         -> the ocular advantage was length leakage

GUARDS: (a) before/after use the SAME truncated rows, differing only in whether
cv/succ come from the full or truncated series, so any change is due to equalization
alone; (b) N sweep with dropout audit, so a collapse cannot be dismissed as small-N
noise; (c) correlation of cv/succ with ORIGINAL epoch count, before vs after --
the leakage account predicts it breaks down together with dAUC.

Zero regeneration: per-epoch ocular_power_ratio is already a column of
X_combined84.csv, and cv/succ are exactly variability_from_ratios() of that series.
Methodology (CV, classifiers, subject aggregation) imported from classify.py.
"""
import sys
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import make_classifiers, evaluate, K, RANDOM_STATE   # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
P, C, S = 'ocular_power_ratio', 'ocular_cv', 'ocular_succ_diff'
ARM_ORDER = ['neural', 'neural+P', 'neural+P+C', 'neural+P+S', 'neural+P+C+S']
PRIMARY_ARM = 'neural+P+C+S'


# ---------------------------------------------------------------- data
def load_combined(path):
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    groups = np.array([row[0] for row in rows])
    y = np.array([int(row[1]) for row in rows])
    feat_names = header[2:]
    X = np.array([[float(v) for v in row[2:]] for row in rows])
    return X, y, groups, feat_names


def arm_indices(feat_names):
    neural = [c for c in feat_names if not c.startswith('ocular_')]
    combos = {
        'neural':       neural,
        'neural+P':     neural + [P],
        'neural+P+C':   neural + [P, C],
        'neural+P+S':   neural + [P, S],
        'neural+P+C+S': neural + [P, C, S],
    }
    return {k: [feat_names.index(c) for c in v] for k, v in combos.items()}


# ------------------------------------------------------- variability
def variability(r):
    """Same definition as features.variability_from_ratios (cv uses ddof=1)."""
    r = np.asarray(r, float).ravel()
    mean = r.mean()
    cv = float(r.std(ddof=1) / mean) if (mean > 0 and len(r) > 1) else 0.0
    succ = float(np.mean(np.abs(np.diff(r)))) if len(r) > 1 else 0.0
    return cv, succ


def pick(n_ep, N, mode):
    """Row offsets (within a subject) kept at size N."""
    if mode == 'first':
        return np.arange(N)
    return np.unique(np.linspace(0, n_ep - 1, N).round().astype(int))


def equalize(X, y, groups, feat_names, N, mode='first'):
    """Truncate to N epochs/subject.

    Returns (X_before, X_after, y2, groups2, kept_subjects, dropped).
      before: truncated rows, cv/succ kept as-is (from the FULL series)
      after : truncated rows, cv/succ RECOMPUTED from the truncated series
    """
    iP, iC, iS = (feat_names.index(c) for c in (P, C, S))
    subs = list(dict.fromkeys(groups.tolist()))
    keep_rows, dropped = [], []
    for s in subs:
        idx = np.where(groups == s)[0]          # epoch order preserved
        if len(idx) < N:
            dropped.append(s)
            continue
        keep_rows.append(idx[pick(len(idx), N, mode)])
    if not keep_rows:
        raise ValueError('no subject has enough epochs')
    rows = np.concatenate(keep_rows)
    Xb = X[rows].copy()
    yb = y[rows]
    gb = groups[rows]
    Xa = Xb.copy()
    for s in dict.fromkeys(gb.tolist()):
        m = np.where(gb == s)[0]
        cv, sc = variability(Xa[m, iP])
        Xa[m, iC] = cv
        Xa[m, iS] = sc
    kept = [s for s in subs if s not in dropped]
    return Xb, Xa, yb, gb, kept, dropped


# ------------------------------------------------------------ stats
def rank(a):
    """Average ranks (ties handled) -- for Spearman without scipy."""
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float('nan'), float('nan')
    pear = float(np.corrcoef(a, b)[0, 1])
    spear = float(np.corrcoef(rank(a), rank(b))[0, 1])
    return pear, spear


def subject_level(Xm, gm, feat_names, col):
    """One value per subject (features are constant within subject for cv/succ)."""
    i = feat_names.index(col)
    subs = list(dict.fromkeys(gm.tolist()))
    return subs, np.array([Xm[gm == s, i][0] for s in subs])


def run_arms(Xm, ym, gm, idx_map, splits):
    out = {}
    clfs = make_classifiers()
    for name in ARM_ORDER:
        Xa = Xm[:, idx_map[name]]
        for cname, make in clfs.items():
            out[(name, cname)] = evaluate(Xa, ym, gm, splits, make)['auc']
    return out, list(clfs)


def d_table(auc, clf_names, title):
    print(f'\n[{title}]')
    print(f'  {"arm":<14}' + ''.join(f'{c:>10}' for c in clf_names))
    for name in ARM_ORDER:
        row = ''.join(f'{np.nanmean(auc[(name, c)]):>10.3f}' for c in clf_names)
        tag = '   <- PRIMARY' if name == PRIMARY_ARM else ''
        print(f'  {name:<14}{row}{tag}')
    print(f'  {"dAUC vs neural":<14}')
    for name in ARM_ORDER:
        if name == 'neural':
            continue
        cells = []
        for c in clf_names:
            d = auc[(name, c)] - auc[('neural', c)]
            cells.append(f'{np.nanmean(d):>+8.3f} {int((d > 0).sum())}/{K}')
        print(f'    {name:<12}' + ''.join(f'{x:>16}' for x in cells))


class _Tee:
    encoding = 'utf-8'

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except (ValueError, OSError):
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except (ValueError, OSError):
                pass


# ------------------------------------------------------------- main
if __name__ == '__main__':
    out_dir = _RESULTS / 'corrected'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'gate5b_equalize.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    X, y, groups, feat_names = load_combined(_RESULTS / 'X_combined84.csv')
    idx_map = arm_indices(feat_names)

    subs = list(dict.fromkeys(groups.tolist()))
    n_ep = np.array([int((groups == s).sum()) for s in subs])
    lab = np.array([int(y[groups == s][0]) for s in subs])
    N_min = int(n_ep.min())

    print('=== 5b confounder gate: epoch equalization ===')
    print('PRIMARY (pre-specified): (neural+P+C+S) - neural, equalized at N=min,')
    print('                         first-N window, AUC, 3 classifiers\n')
    print(f'  subjects={len(subs)}  epochs={len(y)}  K={K}  seed={RANDOM_STATE}')
    print(f'  epochs/subject: min={n_ep.min()} q1={int(np.percentile(n_ep, 25))} '
          f'median={int(np.median(n_ep))} q3={int(np.percentile(n_ep, 75))} max={n_ep.max()}')
    print(f'    ADHD  median={int(np.median(n_ep[lab == 1]))}  '
          f'Control median={int(np.median(n_ep[lab == 0]))}')
    print(f'  epoch-level class prior (before) = {y.mean():.3f}')
    print(f'  N_min = {N_min}  -> no subject dropped\n')

    # ---- primary: N = N_min, first-N
    Xb, Xa, y2, g2, kept, dropped = equalize(X, y, groups, feat_names, N_min, 'first')
    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(np.zeros(len(y2)), y2, g2))
    print(f'  after equalization: subjects={len(kept)} epochs={len(y2)} '
          f'prior={y2.mean():.3f} dropped={len(dropped)}')

    auc_b, clf_names = run_arms(Xb, y2, g2, idx_map, splits)
    auc_a, _ = run_arms(Xa, y2, g2, idx_map, splits)

    d_table(auc_b, clf_names, 'BEFORE equalization (same rows, cv/succ from FULL series)')
    d_table(auc_a, clf_names, 'AFTER equalization (cv/succ recomputed on truncated series)')

    # ---- mechanistic diagnostic
    print('\n[mechanism] correlation of cv / succ_diff with ORIGINAL epoch count')
    print(f'  {"feature":<12}{"before r":>12}{"before rho":>12}{"after r":>12}{"after rho":>12}')
    sub_ids, _ = subject_level(Xb, g2, feat_names, C)
    n_orig = np.array([int((groups == s).sum()) for s in sub_ids])
    for col, tag in ((C, 'cv'), (S, 'succ_diff')):
        _, vb = subject_level(Xb, g2, feat_names, col)
        _, va = subject_level(Xa, g2, feat_names, col)
        pb, sb = corr(vb, n_orig)
        pa, sa = corr(va, n_orig)
        print(f'  {tag:<12}{pb:>12.3f}{sb:>12.3f}{pa:>12.3f}{sa:>12.3f}')
    print('  (leakage account predicts: sizeable before, collapsing after)')

    # ---- sensitivity: N sweep with dropout audit
    print('\n[sensitivity] N sweep, first-N window   '
          '(dAUC of PRIMARY arm vs neural, after equalization)')
    print(f'  {"N":>5}{"kept":>7}{"drpADHD":>9}{"drpCtrl":>9}{"prior":>8}'
          + ''.join(f'{c:>16}' for c in clf_names))
    grid = sorted({N_min, int(np.percentile(n_ep, 10)), int(np.percentile(n_ep, 25)),
                   int(np.median(n_ep))})
    for N in grid:
        if N < 3:
            continue
        Xb2, Xa2, y3, g3, kept2, drop2 = equalize(X, y, groups, feat_names, N, 'first')
        if len(set(y3.tolist())) < 2 or len(kept2) < 10:
            continue
        d_adhd = sum(1 for s in drop2 if y[groups == s][0] == 1)
        d_ctrl = len(drop2) - d_adhd
        sp2 = list(StratifiedGroupKFold(n_splits=K, shuffle=True,
                                        random_state=RANDOM_STATE).split(
            np.zeros(len(y3)), y3, g3))
        a2, _ = run_arms(Xa2, y3, g3, idx_map, sp2)
        cells = []
        for c in clf_names:
            d = a2[(PRIMARY_ARM, c)] - a2[('neural', c)]
            cells.append(f'{np.nanmean(d):>+8.3f} {int((d > 0).sum())}/{K}')
        print(f'  {N:>5}{len(kept2):>7}{d_adhd:>9}{d_ctrl:>9}{y3.mean():>8.3f}'
              + ''.join(f'{x:>16}' for x in cells))

    # ---- sensitivity: evenly-spaced window at N_min
    Xb3, Xa3, y4, g4, _, _ = equalize(X, y, groups, feat_names, N_min, 'even')
    sp3 = list(StratifiedGroupKFold(n_splits=K, shuffle=True,
                                    random_state=RANDOM_STATE).split(
        np.zeros(len(y4)), y4, g4))
    a3, _ = run_arms(Xa3, y4, g4, idx_map, sp3)
    print(f'\n[sensitivity] evenly-spaced window, N={N_min}  '
          '(dAUC of PRIMARY arm vs neural)')
    cells = []
    for c in clf_names:
        d = a3[(PRIMARY_ARM, c)] - a3[('neural', c)]
        cells.append(f'{np.nanmean(d):>+8.3f} {int((d > 0).sum())}/{K}')
    print('       ' + ''.join(f'{x:>16}' for x in cells))

    # ---- verdict on the pre-specified primary
    prim = [np.nanmean(auc_a[(PRIMARY_ARM, c)] - auc_a[('neural', c)]) for c in clf_names]
    prim_b = [np.nanmean(auc_b[(PRIMARY_ARM, c)] - auc_b[('neural', c)]) for c in clf_names]
    n_pos = sum(v > 0 for v in prim)
    print('\n[PRIMARY VERDICT]  (neural+P+C+S) - neural, equalized at N=N_min')
    print('  before: ' + ' '.join(f'{v:+.3f}' for v in prim_b))
    print('  after : ' + ' '.join(f'{v:+.3f}' for v in prim))
    if n_pos == 3:
        print('  SURVIVES 3/3 -> ocular contributes with epoch count physically equalized.')
        print('           Length cannot be the pathway; strongest form of the hypothesis.')
    elif n_pos == 0:
        print('  COLLAPSES 0/3 -> ocular advantage did not survive equalization;')
        print('           consistent with length leakage through variability features.')
    else:
        print(f'  PARTIAL {n_pos}/3 -> report the shrinkage as-is; stage-10 bootstrap CI decides.')
    print('  Note: pre-specified contrast; all arms and sensitivities reported regardless.')
    print('  Caution: small N makes cv/succ noisy for everyone (noise, not bias) --')
    print('           read the N sweep before concluding that a collapse is real.')

    sys.stdout = orig
    fh.close()