"""Stage 7 feature enrichment: channel-level, nonlinear and connectivity features.

Governed by PREREG.md section 3.6 (feature set fixed a priori, no post-hoc addition
or removal) and 3.7 (blinding: only neural(+length) performance may be inspected
until the feature set is frozen).

FEATURE SET (306 per epoch, all channel- or region-level)
  spectral   19 ch x 4 bands relative-power CLR                          =  76
  Hjorth     variance(activity), hjorth_mobility, hjorth_complexity      =  57
  complexity higuchi_fd, katz_fd, samp_entropy, spect_entropy            =  76
  statistics kurtosis, skewness, std                                     =  57
  connectivity  10 region pairs x 4 bands, magnitude-squared coherence   =  40

TBR is excluded from the model set: log(TBR) = clr_theta - clr_beta exactly, so it
is mathematically redundant with the CLR block (verified numerically in stage 3b).

Coherence is not provided by mne-features, so it is computed with scipy.signal
exactly as PREREG specifies (region pairs x bands). This is an implementation
choice, not a deviation from the pre-registered feature set.

CACHE LAYER
  neural_source() runs bandpass -> CAR -> ASR -> ICA -> ICLabel -> back-projection,
  which dominates runtime. Stage 3b showed that changing a feature definition
  invalidates the old feature cache and forces a full re-run over 121 subjects.
  This module therefore caches the back-projected 19-channel EPOCHS themselves
  (exploration/_cache/brain_{name}.npz, float32). ICA then runs once; any later
  change to feature definitions costs only the feature pass.

USAGE
  python src/features_ext.py            # all subjects
  python src/features_ext.py --limit 5  # smoke test on the first 5 subjects
Output: results/features_neural_ext.csv  (subject_id, label, rec_length_sec, 306 features)
"""
import sys
import csv
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import scipy.signal

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from load_data import all_subjects, load_subject, SFREQ, CH          # noqa: E402
from features import (neural_source, make_epochs_2d, psd_fft, band_power,  # noqa: E402
                      clr, BAND_RANGES, BAND_NAMES, REGIONS, REGION_IDX,
                      EPOCH_SEC, _CACHE, _RESULTS)
from mne_features.feature_extraction import extract_features          # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

UNIVARIATE = ['variance', 'hjorth_mobility', 'hjorth_complexity',
              'higuchi_fd', 'katz_fd', 'samp_entropy', 'spect_entropy',
              'kurtosis', 'skewness', 'std']
REGION_PAIRS = list(combinations(list(REGIONS), 2))          # 10 pairs
_OUT_CSV = _RESULTS / 'features_neural_ext.csv'


def column_names():
    cols = [f'{ch}_{b}_clr' for ch in CH for b in BAND_NAMES]          # 76
    cols += [f'{fn}_{ch}' for fn in UNIVARIATE for ch in CH]           # 190
    cols += [f'{a}-{b}_{bn}_coh' for a, b in REGION_PAIRS
             for bn in BAND_NAMES]                                     # 40
    return cols


# ------------------------------------------------------------ cache
def brain_epochs(name, path):
    """Back-projected brain epochs (n_ep, 19, n_times), cached.

    Returns dict(epochs, n_brain, eligible). ICA runs only on a cache miss.
    """
    cache = _CACHE / f'brain_{name}.npz'
    if cache.exists():
        d = np.load(cache)
        return {'epochs': d['epochs'], 'n_brain': int(d['n_brain']),
                'eligible': bool(d['eligible'])}
    data, n_brain, eligible = neural_source(path)
    if not eligible:
        ep = np.zeros((0, len(CH), int(round(SFREQ * EPOCH_SEC))), dtype=np.float32)
    else:
        chunks = make_epochs_2d(data)
        ep = (np.stack(chunks).astype(np.float32) if chunks
              else np.zeros((0, len(CH), int(round(SFREQ * EPOCH_SEC))), dtype=np.float32))
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, epochs=ep, n_brain=np.int64(n_brain),
                        eligible=np.bool_(eligible))
    return {'epochs': ep, 'n_brain': n_brain, 'eligible': eligible}


# ------------------------------------------------------- feature blocks
def spectral_clr(epochs):
    """(n_ep, 19, T) -> (n_ep, 76). Per channel: 4-band relative power -> CLR."""
    out = np.empty((epochs.shape[0], len(CH) * len(BAND_NAMES)), float)
    for i, ep in enumerate(epochs):
        vals = []
        for c in range(len(CH)):
            f, pxx = psd_fft(ep[c])
            bp = np.array([band_power(f, pxx, lo, hi) for lo, hi in BAND_RANGES])
            tot = bp.sum()
            rel = bp / tot if tot > 0 else np.full(len(BAND_RANGES), 1.0 / len(BAND_RANGES))
            vals.extend(clr(rel).tolist())
        out[i] = vals
    return out


def region_coherence(epochs, fs=SFREQ):
    """(n_ep, 19, T) -> (n_ep, 40). Magnitude-squared coherence between region-mean
    signals, averaged within each band."""
    n_ep = epochs.shape[0]
    out = np.empty((n_ep, len(REGION_PAIRS) * len(BAND_NAMES)), float)
    nperseg = min(256, epochs.shape[2])
    for i, ep in enumerate(epochs):
        reg = {r: ep[idx].mean(axis=0) for r, idx in REGION_IDX.items()}
        vals = []
        for a, b in REGION_PAIRS:
            f, cxy = scipy.signal.coherence(reg[a], reg[b], fs=fs, nperseg=nperseg)
            for lo, hi in BAND_RANGES:
                m = (f >= lo) & (f <= hi)
                vals.append(float(cxy[m].mean()) if m.any() else 0.0)
        out[i] = vals
    return out


def subject_features(epochs):
    """(n_ep, 19, T) -> (n_ep, 306) in the order of column_names()."""
    if epochs.shape[0] == 0:
        return np.zeros((0, len(column_names())), float)
    sp = spectral_clr(epochs)
    uni = extract_features(epochs.astype(float), SFREQ, UNIVARIATE)
    coh = region_coherence(epochs)
    return np.hstack([sp, uni, coh])


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])

    subjects = all_subjects()
    if limit:
        subjects = subjects[:limit]

    cols = column_names()
    print('=== stage 7 extended neural features ===')
    print(f'  subjects={len(subjects)}  features={len(cols)} '
          f'(clr 76 + univariate {len(UNIVARIATE) * len(CH)} + coh {len(REGION_PAIRS) * 4})')
    print(f'  cache dir: {_CACHE}\n')

    rows, n_ok, t0 = [], 0, time.time()
    for i, (path, label) in enumerate(subjects, 1):
        name = path.stem
        hit = (_CACHE / f'brain_{name}.npz').exists()
        r = brain_epochs(name, path)
        ep = r['epochs']
        if not r['eligible'] or ep.shape[0] == 0:
            print(f'[{i:3d}/{len(subjects)}] {name:<7} skipped (no brain IC)')
            continue
        n_ok += 1
        length_sec = load_subject(path).shape[0] / SFREQ
        F = subject_features(ep)
        for v in F:
            rows.append([name, int(label), round(length_sec, 3), *[f'{x:.6g}' for x in v]])
        print(f'[{i:3d}/{len(subjects)}] {name:<7} epochs={ep.shape[0]:>3} '
              f'brainIC={r["n_brain"]:>2} {"cache" if hit else "ICA "} '
              f'{time.time() - t0:6.1f}s')

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'label', 'rec_length_sec', *cols])
        w.writerows(rows)

    arr = np.array([[float(v) for v in r[3:]] for r in rows], float) if rows else np.zeros((0, len(cols)))
    print(f'\n[check] subjects with brain IC={n_ok}  epochs={len(rows)}  '
          f'cols={len(cols)}  NaN={int(np.isnan(arr).sum())}  '
          f'Inf={int(np.isinf(arr).sum())}')
    if len(rows):
        clr_block = arr[:, :len(CH) * len(BAND_NAMES)].reshape(len(rows), len(CH), len(BAND_NAMES))
        s = clr_block.sum(axis=2)
        print(f'  CLR sum per channel: {s.min():.3f}~{s.max():.3f} (0 expected)')
        coh_block = arr[:, -len(REGION_PAIRS) * len(BAND_NAMES):]
        print(f'  coherence range: {coh_block.min():.3f}~{coh_block.max():.3f} (0..1 expected)')
    print(f'  saved: {_OUT_CSV}')
    print(f'  elapsed: {time.time() - t0:.1f}s')