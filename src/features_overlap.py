"""Stage 2: 6-second epochs at 50% overlap (PREREG 8.3(6), 8.3.3(3)).

Exploratory arm. The pre-registered primary analysis stays on 6-second
non-overlapping epochs and is not touched.

WHY THIS IS CHEAP
  exploration/_cache/brain_{subject}.npz already holds the brain-only epochs,
  and make_epochs_2d cuts contiguously from t=0 discarding only the tail, so
  concatenating those epochs along time restores the continuous back-projected
  signal exactly. Re-cutting it at a different stride needs no ICA.

WHY THE NON-OVERLAP ARM IS REBUILT HERE TOO
  The control must come from the same reassembled signal, otherwise the
  comparison carries two differences instead of one. It also validates the
  reassembly for free: ov00 must reproduce results/conditions/features_i_brain.csv.

HOW THE REASSEMBLY CHECK IS SCORED
  Every epoch of every subject is compared, not just the first, and each of the
  306 features is scored in units of ITS OWN spread:

      deviation_j = max_i |ov00[i, j] - stage4[i, j]| / std_j(stage4)

  An absolute difference is meaningless across this feature set, where kurtosis
  runs to tens and coherence is bounded by one. Dividing by the column standard
  deviation puts every feature on the same ruler.

  No pass/fail threshold is applied. A hard gate on the worst of 306 features
  fires almost by construction, and picking the threshold after seeing the
  numbers would be exactly the practice this project exists to criticise. The
  magnitudes are reported and translated into their effect on the classifier;
  the decision-relevant check is the AUC comparison in contrast_overlap.py,
  where ov00 must reproduce the stage 4 i_brain AUC.

  Note also that this comparison spans two separate pipeline executions, so it
  carries any run-to-run non-determinism in ASR and ICA on top of CSV rounding.
  It is not a pure test of the reassembly. The ov00/ov50 contrast itself is
  unaffected: both arms are cut from one and the same restored signal.

WHAT STAGE 2 DOES AND DOES NOT ANSWER
  The prior study used 2-second epochs at 50% overlap. Only the overlap half is
  testable here: 2 seconds is 256 samples, and region coherence uses nperseg=256,
  which would leave a single window and make coherence identically 1. Changing
  that would require redefining features, which PREREG 3.6 forbids. So a null
  result here means "overlap alone does not close the gap", NOT "epoching is
  not the cause" (PREREG 8.3(5)).

PROMOTION RULE (fixed in PREREG 8.3(6) before this ran)
  If the neural-only arm gains dAUC >= +0.03 on ALL THREE tier samples
  (full121 / tier1_91 / tier2_86), overlap is promoted to the primary design in
  a separate deviation entry. Otherwise it is reported as exploratory only.
  Judging on the neural-only arm keeps the ocular contrast unseen at decision
  time, the same blinding structure as PREREG 3.7.

OUTPUT
  results/overlap/features_i_brain_ov00.csv   non-overlap control, rebuilt
  results/overlap/features_i_brain_ov50.csv   50% overlap
  results/overlap/extract_log.txt

USAGE
  C:\\envs\\eeg\\python.exe src/features_overlap.py --limit 5
  C:\\envs\\eeg\\python.exe src/features_overlap.py
"""
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from load_data import all_subjects, load_subject, SFREQ                # noqa: E402
from features import EPOCH_SEC, _CACHE, _RESULTS                       # noqa: E402
from features_ext import subject_features, column_names                # noqa: E402
from features_cond import _Tee                                         # noqa: E402

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_OUT_DIR = _RESULTS / 'overlap'
_REF_CSV = _RESULTS / 'conditions' / 'features_i_brain.csv'
OVERLAP = 0.5                       # PREREG 8.3(6): single value, no sweep


def cut(data, stride):
    """Slice (channels, time) into fixed-length windows advancing by `stride`."""
    n = int(round(SFREQ * EPOCH_SEC))
    out, start = [], 0
    while start + n <= data.shape[1]:
        out.append(data[:, start:start + n])
        start += stride
    return out


def brain_signal(name):
    """Continuous brain-only signal, restored from the cached non-overlap epochs."""
    cache = _CACHE / ('brain_%s.npz' % name)
    if not cache.exists():
        return None, 'no cache'
    d = np.load(cache)
    if not bool(d['eligible']) or d['epochs'].shape[0] == 0:
        return None, 'no brain IC'
    return np.concatenate(list(d['epochs']), axis=-1).astype(np.float64), None


def load_reference():
    """Full stage 4 i_brain feature table: {subject: (n_ep, 306)}, plus column scales.

    Column scales are the standard deviations over the whole table; they are the
    ruler the reassembly check is measured against.
    """
    if not _REF_CSV.exists():
        return None, None
    ref = {}
    with open(_REF_CSV, newline='') as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            ref.setdefault(row[0], []).append([float(v) for v in row[3:]])
    ref = {k: np.array(v, float) for k, v in ref.items()}
    allrows = np.vstack(list(ref.values()))
    scale = allrows.std(axis=0)
    scale[scale == 0] = 1.0                  # constant columns: fall back to absolute
    return ref, scale


def main(limit):
    subjects = all_subjects()
    if limit:
        subjects = subjects[:limit]
    cols = column_names()
    n = int(round(SFREQ * EPOCH_SEC))
    stride50 = n // 2

    print('=' * 78)
    print('STAGE 2 -- 6 s epochs at 50%% overlap (PREREG 8.3(6))')
    print('=' * 78)
    print('epoch      : %.0f s = %d samples' % (EPOCH_SEC, n))
    print('stride     : %d (non-overlap) / %d (50%% overlap)' % (n, stride50))
    print('signal     : brain-only, restored from exploration/_cache/brain_*.npz')
    print('features   : %d, definitions frozen at features-v1' % len(cols))
    print('subjects   : %d\n' % len(subjects))

    ref, scale = load_reference()
    rows00, rows50, t0 = [], [], time.time()
    devs = np.zeros(len(cols)) if ref else None
    n_cmp = 0

    for i, (path, label) in enumerate(subjects, 1):
        name = path.stem
        data, why = brain_signal(name)
        if data is None:
            print('[%3d/%3d] %-8s skipped (%s)' % (i, len(subjects), name, why))
            continue
        length_sec = load_subject(path).shape[0] / SFREQ

        e00, e50 = cut(data, n), cut(data, stride50)
        F00 = subject_features(np.stack(e00).astype(np.float32))
        F50 = subject_features(np.stack(e50).astype(np.float32))

        for v in F00:
            rows00.append([name, int(label), round(length_sec, 3),
                           *['%.6g' % x for x in v]])
        for v in F50:
            rows50.append([name, int(label), round(length_sec, 3),
                           *['%.6g' % x for x in v]])

        dev = ''
        if ref is not None and name in ref and ref[name].shape == F00.shape:
            d = np.abs(F00 - ref[name]).max(axis=0) / scale
            devs = np.maximum(devs, d)
            n_cmp += 1
            dev = '  dev %.1e' % float(d.max())
        elif ref is not None and name in ref:
            dev = '  SHAPE MISMATCH %s vs %s' % (F00.shape, ref[name].shape)
        print('[%3d/%3d] %-8s ep %3d -> %3d%s   %5.1fs'
              % (i, len(subjects), name, len(e00), len(e50), dev, time.time() - t0))
        sys.stdout.flush()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, rows in (('ov00', rows00), ('ov50', rows50)):
        out = _OUT_DIR / ('features_i_brain_%s.csv' % tag)
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['subject_id', 'label', 'rec_length_sec', *cols])
            w.writerows(rows)
        arr = np.array([[float(v) for v in r[3:]] for r in rows], float)
        print('\n  %-5s subjects=%3d epochs=%5d NaN=%d Inf=%d  %s'
              % (tag, len(set(r[0] for r in rows)), len(rows),
                 int(np.isnan(arr).sum()), int(np.isinf(arr).sum()), out.name))

    print('\n' + '-' * 78)
    print('REASSEMBLY CHECK -- ov00 against the stage 4 i_brain table')
    print('-' * 78)
    if ref is None:
        print('  skipped: %s not found' % _REF_CSV)
    elif n_cmp == 0:
        print('  no subject could be compared')
    else:
        order = np.argsort(-devs)
        print('  subjects compared : %d, all epochs, all %d features' % (n_cmp, len(cols)))
        print('  deviation is max|difference| divided by that feature\'s own std')
        print('  median %.1e   p99 %.1e   max %.1e'
              % (float(np.median(devs)), float(np.percentile(devs, 99)),
                 float(devs.max())))
        print('  features above 1e-4 : %d of %d' % (int((devs > 1e-4).sum()), len(cols)))
        print('  worst five:')
        for j in order[:5]:
            print('    %-28s %.2e' % (cols[j], devs[j]))
        worst = float(devs.max())
        print('\n  what this means for the classifier:')
        print('    features are standardised before fitting, so a deviation of %.1e'
              % worst)
        print('    is a shift of %.1e in standardised units. With logistic'
              % worst)
        print('    coefficients of order 0.1 to 1 that moves the logit by about')
        print('    %.0e and a predicted probability by less than that.' % worst)
        print('\n  caveat: this spans two pipeline executions, so it includes any')
        print('  run-to-run difference in ASR and ICA, not only rounding. Moment')
        print('  based features (variance, std, kurtosis) surface such differences')
        print('  first because they are driven by extreme samples.')
        print('  The ov00/ov50 contrast is not affected: both are cut from the same')
        print('  restored signal. The decisive check is the AUC comparison in')
        print('  contrast_overlap.py.')
    print('epoch inflation: %d -> %d rows (expected about 2x - 1 per subject)'
          % (len(rows00), len(rows50)))
    print('elapsed: %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    with open(_OUT_DIR / 'extract_log.txt', 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            main(args.limit)
        finally:
            sys.stdout = orig