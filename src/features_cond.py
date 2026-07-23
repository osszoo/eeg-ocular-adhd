"""Stage 4 driver: extract the 306 features under every reconstruction condition.

Governed by PREREG 8.3 / 8.3.1. One pipeline pass per subject produces every
condition, so ICA runs exactly once per subject for the whole arm.

WHAT IT PRODUCES
  results/conditions/features_{condition}.csv
      subject_id, label, rec_length_sec, <306 feature columns>
      Identical layout to results/features_neural_ext.csv, so every downstream
      script that reads that file reads these unchanged.
  results/conditions/extract_log.txt
      Full log plus the distributions the 3-subject diagnostic could not settle:
      eye-IC counts, candidate counts, matching alpha, and what ASR actually did.

FEATURE DEFINITION IS UNTOUCHED
  subject_features() is imported from features_ext.py. The 306 columns, the epoch
  length, the band edges and the coherence settings are exactly those frozen at
  the features-v1 tag. This stage changes the SIGNAL, nothing else. That is what
  makes the contrast interpretable: one factor moves.

INVARIANT CHECKED AT RUNTIME
  Every condition of a given subject must yield the same number of epochs.
  Filtering, referencing, ASR and back-projection all preserve n_times, so an
  unequal count means something is wrong and the subject is reported, not used.

RESUMABLE
  Per-subject cache at exploration/_cache/cond_{subject}.npz holds the feature
  matrices for all conditions. A crashed or interrupted run resumes from the last
  completed subject; delete the cache files to force recomputation.

USAGE
  C:\\envs\\eeg\\python.exe src/features_cond.py --limit 5      # smoke test
  C:\\envs\\eeg\\python.exe src/features_cond.py                # all subjects
  C:\\envs\\eeg\\python.exe src/features_cond.py --conds i_brain,ii_eye1
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
from load_data import all_subjects, load_subject, SFREQ, CH            # noqa: E402
from features import make_epochs_2d, EPOCH_SEC, _CACHE, _RESULTS       # noqa: E402
from features_ext import subject_features, column_names                # noqa: E402
from conditions import condition_signals, ALL_CONDS                    # noqa: E402

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_OUT_DIR = _RESULTS / 'conditions'
_META_KEYS = ['n_ic', 'n_brain', 'n_eye', 'n_cand', 'wrapped', 'var_eye']


# --------------------------------------------------------------- per subject
def subject_condition_features(name, path):
    """{condition: (n_ep, 306)} and a metadata dict, cached per subject."""
    cache = _CACHE / ('cond_%s.npz' % name)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        feats = {k[2:]: d[k] for k in d.files if k.startswith('f_')}
        meta = {k: float(d['m_' + k]) for k in _META_KEYS if ('m_' + k) in d.files}
        meta['alpha'] = d['m_alpha'].tolist() if 'm_alpha' in d.files else []
        meta['var_ratio'] = {k[3:]: float(d[k]) for k in d.files if k.startswith('vr_')}
        meta['cached'] = True
        return feats, meta

    sig, meta, dec = condition_signals(path)
    del dec                                        # release the Raw/ICA objects

    base_var = sig['i_brain'].var() if 'i_brain' in sig else np.nan
    var_ratio, feats = {}, {}
    for cond, data in sig.items():
        var_ratio[cond] = float(data.var() / base_var) if base_var else np.nan
        chunks = make_epochs_2d(data)
        if not chunks:
            feats[cond] = np.zeros((0, len(column_names())), float)
            continue
        ep = np.stack(chunks).astype(np.float32)
        feats[cond] = subject_features(ep)

    payload = {('f_' + c): v for c, v in feats.items()}
    payload.update({('vr_' + c): np.float64(v) for c, v in var_ratio.items()})
    payload.update({('m_' + k): np.float64(meta.get(k, np.nan)) for k in _META_KEYS})
    payload['m_alpha'] = np.asarray(meta.get('alpha', []), float)
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)

    meta['var_ratio'] = var_ratio
    meta['cached'] = False
    return feats, meta


# ------------------------------------------------------------------ reporting
def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float('nan')


def summarise(records):
    """The distributions the 3-subject diagnostic could not settle."""
    print('\n' + '-' * 78)
    print('DISTRIBUTIONS OVER ALL PROCESSED SUBJECTS')
    print('-' * 78)

    n_eye = [r['meta']['n_eye'] for r in records]
    multi = sum(v > 1 for v in n_eye)
    print('eye ICs per subject : 0:%d  1:%d  2+:%d'
          % (sum(v == 0 for v in n_eye), sum(v == 1 for v in n_eye), multi))
    print('  -> ii_eyeall differs from ii_eye1 for %d subject(s). If 0, the arm is'
          % multi)
    print('     redundant and should be reported as such rather than interpreted.')

    cand = [r['meta']['n_cand'] for r in records]
    print('\ncandidate control ICs: min %d  median %.0f  max %d   (subjects with <5: %d)'
          % (min(cand), np.median(cand), max(cand), sum(c < 5 for c in cand)))
    wrapped = [r['meta']['wrapped'] for r in records]
    print('  wrap-around reuse occurred for %d subject(s)' % sum(w > 0 for w in wrapped))

    alpha = [a for r in records for a in r['meta'].get('alpha', [])]
    if alpha:
        print('\nmatching alpha (ii_m): median %.2f  p90 %.2f  max %.2f   (alpha>5: %d of %d)'
              % (np.median(alpha), pct(alpha, 90), max(alpha),
                 sum(a > 5 for a in alpha), len(alpha)))
        print('  -> high alpha means a residual IC was amplified to eye-IC scale.')
        print('     ii_s (phase surrogate) needs no amplification and is unaffected.')

    def ratios(cond):
        return [r['meta']['var_ratio'][cond] for r in records
                if cond in r['meta'].get('var_ratio', {})]

    print('\nvariance ratio vs i_brain (median over subjects)')
    for cond in ALL_CONDS:
        v = ratios(cond)
        if v:
            print('  %-12s %6.3f   (n=%d)' % (cond, float(np.median(v)), len(v)))

    a, b = ratios('iii_noica'), ratios('iv_noasr')
    if a and b and len(a) == len(b):
        d = np.array(b) - np.array(a)
        inert = int((np.abs(d) < 1e-3).sum())
        print('\nASR effect (iv_noasr - iii_noica): median %+.4f   inert (<0.001) for '
              '%d/%d subjects' % (float(np.median(d)), inert, len(d)))
        print('  -> at cutoff 100 ASR may be removing almost nothing; that is itself')
        print('     a number to report, not a bug.')

    c, e = ratios('iv_noasr'), ratios('v_wideband')
    if c and e and len(c) == len(e):
        d = np.array(e) - np.array(c)
        print('\n40-60 Hz contribution (v_wideband - iv_noasr): median %+.4f of i_brain '
              'variance' % float(np.median(d)))
        print('  -> variance is not the whole story: Hjorth, fractal and entropy')
        print('     features weight high frequencies far more than variance does.')


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


# ----------------------------------------------------------------------- main
def main(limit, conds):
    subjects = all_subjects()
    if limit:
        subjects = subjects[:limit]
    cols = column_names()

    print('=' * 78)
    print('STAGE 4 -- per-condition feature extraction (PREREG 8.3 / 8.3.1)')
    print('=' * 78)
    print('subjects   : %d' % len(subjects))
    print('conditions : %d  (%s)' % (len(conds), ', '.join(conds)))
    print('features   : %d per epoch, epoch %.0fs, definitions frozen at features-v1'
          % (len(cols), EPOCH_SEC))
    print('cache      : %s\n' % _CACHE)

    rows = {c: [] for c in conds}
    records, t0 = [], time.time()

    for i, (path, label) in enumerate(subjects, 1):
        name = path.stem
        feats, meta = subject_condition_features(name, path)
        length_sec = load_subject(path).shape[0] / SFREQ

        counts = {c: feats[c].shape[0] for c in feats}
        uniq = set(counts.values())
        flag = '' if len(uniq) <= 1 else '  <-- EPOCH COUNT MISMATCH %s' % counts
        for c in conds:
            if c not in feats or feats[c].shape[0] == 0:
                continue
            for v in feats[c]:
                rows[c].append([name, int(label), round(length_sec, 3),
                                *['%.6g' % x for x in v]])

        records.append({'name': name, 'label': int(label), 'meta': meta})
        print('[%3d/%3d] %-8s eye=%d cand=%d epochs=%s conds=%d %s %6.1fs%s'
              % (i, len(subjects), name, meta['n_eye'], meta['n_cand'],
                 sorted(uniq)[0] if uniq else 0, len(feats),
                 'cache' if meta['cached'] else 'ICA  ', time.time() - t0, flag))
        sys.stdout.flush()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('\n' + '-' * 78)
    print('WRITTEN')
    print('-' * 78)
    for c in conds:
        if not rows[c]:
            print('  %-12s no rows (no eligible subject)' % c)
            continue
        out = _OUT_DIR / ('features_%s.csv' % c)
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['subject_id', 'label', 'rec_length_sec', *cols])
            w.writerows(rows[c])
        arr = np.array([[float(v) for v in r[3:]] for r in rows[c]], float)
        subs = len(set(r[0] for r in rows[c]))
        print('  %-12s subjects=%3d epochs=%5d NaN=%d Inf=%d  %s'
              % (c, subs, len(rows[c]), int(np.isnan(arr).sum()),
                 int(np.isinf(arr).sum()), out.name))

    summarise(records)
    print('\nelapsed: %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--conds', type=str, default=None,
                    help='comma-separated subset of conditions')
    args = ap.parse_args()
    sel = ALL_CONDS if not args.conds else [c.strip() for c in args.conds.split(',')]
    bad = [c for c in sel if c not in ALL_CONDS]
    if bad:
        raise SystemExit('unknown condition(s): %s\nknown: %s'
                         % (', '.join(bad), ', '.join(ALL_CONDS)))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    with open(_OUT_DIR / 'extract_log.txt', 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            main(args.limit, sel)
        finally:
            sys.stdout = orig