"""Stage 5a confounder gate: variability-feature ladder on a length-robust base.

Stage 4 showed recording-length alone reaches AUC 0.68, above the combined pipeline.
So: is the combined advantage a real ocular signal, or leakage of recording length
through the variability features (cv, succ_diff)? Those two are computed from the
per-epoch power-ratio time series, whose length (= epoch count, group-correlated)
can seep in. power_ratio (an epoch average) is comparatively length-robust.

Ladder, built up from the length-robust base (neural + P):
  neural
  neural+P        <- PRE-REGISTERED PRIMARY combined (length-robust ocular only)
  neural+P+C
  neural+P+S
  neural+P+C+S    (= current combined baseline)

PRE-SPECIFIED PRIMARY CONTRAST (fixed before seeing results):
  dAUC = AUC(neural+P) - AUC(neural), all 3 classifiers, AUC metric.
  Consistently positive => length-independent ocular signal exists (gate 1 passed).
  Whether C/S add real signal on top of P is decided by 5b (epoch equalization).

Methodology is imported from classify.py (identical CV, classifiers, aggregation);
only the feature columns differ. Zero cost: all 3 ocular features already in
X_combined84.csv, so no regeneration and no cache invalidation.
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


def load_combined(path):
    """X_combined84.csv -> X, y, groups, feat_names."""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    groups = np.array([row[0] for row in rows])
    y = np.array([int(row[1]) for row in rows])
    feat_names = header[2:]
    X = np.array([[float(v) for v in row[2:]] for row in rows])
    return X, y, groups, feat_names


def build_arms(feat_names):
    """arm name -> column indices. neural = every column not starting with ocular_."""
    neural = [c for c in feat_names if not c.startswith('ocular_')]
    combos = {
        'neural':       neural,
        'neural+P':     neural + [P],
        'neural+P+C':   neural + [P, C],
        'neural+P+S':   neural + [P, S],
        'neural+P+C+S': neural + [P, C, S],
    }
    return {name: [feat_names.index(c) for c in cols] for name, cols in combos.items()}


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


if __name__ == '__main__':
    out_dir = _RESULTS / 'corrected'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'gate5a_variability.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    X, y, groups, feat_names = load_combined(_RESULTS / 'X_combined84.csv')
    arms = build_arms(feat_names)
    clfs = make_classifiers()
    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups))

    auc = {}
    for name, idx in arms.items():
        Xa = X[:, idx]
        for cname, make in clfs.items():
            auc[(name, cname)] = evaluate(Xa, y, groups, splits, make)['auc']

    print('=== 5a confounder gate: variability ladder (length-robust base) ===')
    print('PRIMARY (pre-specified): (neural+P) - neural, AUC, consistency over 3 clfs')
    print(f'   n_subjects={len(set(groups.tolist()))}  n_epochs={len(y)}  K={K}  seed={RANDOM_STATE}\n')

    print('[AUC by arm]')
    print(f'  {"arm":<14}{"RF":>10}{"Logistic":>10}{"SVM":>10}')
    for name in arms:
        tag = '   <- PRIMARY' if name == 'neural+P' else ''
        row = ''.join(f'{np.nanmean(auc[(name, c)]):>10.3f}' for c in clfs)
        print(f'  {name:<14}{row}{tag}')

    print('\n[dAUC vs neural   (mean . positive folds)]')
    print(f'  {"arm":<14}{"RF":>16}{"Logistic":>16}{"SVM":>16}')
    for name in arms:
        if name == 'neural':
            continue
        cells = []
        for c in clfs:
            d = auc[(name, c)] - auc[('neural', c)]
            cells.append(f'{np.nanmean(d):>+8.3f} {int((d > 0).sum())}/{K}')
        star = ' *' if name == 'neural+P' else '  '
        print(f'  {name:<14}{star}' + ''.join(f'{x:>16}' for x in cells))

    prim = [np.nanmean(auc[('neural+P', c)] - auc[('neural', c)]) for c in clfs]
    n_pos = sum(v > 0 for v in prim)
    print('\n[PRIMARY VERDICT]  (neural+P) - neural')
    if n_pos == 3:
        print('  PASS 3/3 classifiers positive -> length-robust ocular signal exists.')
        print('       Whether C/S add real signal on top of P is decided by 5b.')
    else:
        print(f'  FAIL {n_pos}/3 classifiers positive -> length-robust ocular signal insufficient;')
        print('       ocular advantage may depend on variability (the length pathway). 5b decides.')
    print('  Note: pre-specified contrast, so this verdict is free of selection effects.')
    print('        All arms reported regardless of outcome.')

    sys.stdout = orig
    fh.close()
