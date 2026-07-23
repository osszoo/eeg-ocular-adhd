"""Stage 4 (branch: refinement) -- reconstruction conditions for the IC-removal arm.

Governed by PREREG 8.3. This module produces, for ONE subject, the 19-channel
signal under every reconstruction condition, running the expensive part of the
pipeline exactly once.

WHY
  The pre-registered primary contrast represented ocular information with three
  summary statistics (power_ratio, cv, succ_diff) and came out null. That leaves
  an alternative explanation open: three numbers may be too coarse a description
  of an ocular independent component. This arm keeps the ocular IC in the SIGNAL
  and measures it with the same 306 channel-level features used everywhere else.
  The same machinery also yields a number nobody has reported -- how many points
  of accuracy artifact removal itself costs.

CONDITIONS
  Hypothesis branch (one ICA fit, back-projection target differs)
    i_brain      brain-qualified ICs only                        <- current baseline
    ii_eye1      brain + top-1 eye IC                            <- PRIMARY contrast
    ii_eyeall    brain + every qualified eye IC                  <- secondary
    ii_c1..c5    brain + k-th control IC, as-is                   <- control
  ii_m1..m5    brain + k-th control IC, variance-matched        <- control
  ii_s1..s3    brain + phase-randomised eye surrogate           <- tightest control

  Ablation ladder (one factor per rung, no ICA)
    iii_noica    0.5-40 Hz, ASR on,  no ICA
    iv_noasr     0.5-40 Hz, ASR off, no ICA
    v_wideband   0.5-60 Hz + 50 Hz notch, ASR off, no ICA   <- prior study's pipeline

CONTROLS (PREREG 8.3.1)
  A first diagnostic showed the discarded ICs are far smaller than the eye IC
  (contribution ratio 0.08-0.61), because brain takes 12-14 of 18 components and
  only 3-5 residual candidates remain. An under-powered control biases IN FAVOUR
  of the hypothesis, so three control families are used, all reported:

    ii_c1..c5  candidate IC as-is, ranked by |V_j - V_eye|      (PREREG 8.3)
    ii_m1..m5  same IC rescaled so its contribution equals V_eye (variance-matched)
    ii_s1..s3  phase-randomised surrogate of the EYE contribution itself:
               identical topography, identical per-channel PSD, identical variance,
               temporal structure destroyed. The tightest possible control.

  Contributions are measured exactly as a back-projection difference,
      contribution_j = apply(keep = brain + j) - apply(keep = brain),
  which carries MNE's internal PCA/mean handling rather than approximating it.
  When a subject has fewer candidates than control arms, assignment wraps around
  so that every arm runs on the same subjects.

ORDER OF OPERATIONS FOR v_wideband
  The prior study applied CAR -> 50 Hz notch -> 0.5-60 Hz bandpass. All three are
  linear, so the order is interchangeable; this module filters then re-references,
  matching the house pipeline, and the result is mathematically identical.

NO CACHING HERE
  This module is a pure function of the input file. Caching belongs to the caller
  (features_cond.py caches the extracted feature rows, which are small and make
  a crashed run resumable). One call = one full pipeline pass = one ICA fit.

USAGE
  C:\\envs\\eeg\\python.exe src/conditions.py --limit 3      # diagnostic
"""
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import mne

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from load_data import load_subject, to_raw, all_subjects            # noqa: E402
from preprocess import bandpass, car, asr, ica, L_FREQ, H_FREQ      # noqa: E402
from features import EYE, BRAIN, TAU                                # noqa: E402
from mne_icalabel.iclabel import iclabel_label_components           # noqa: E402

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

WIDE_HFREQ = 60.0          # prior study's lowpass; feasible at 128 Hz (Nyquist 64)
NOTCH_HZ = 50.0            # mains frequency of the recording site
N_CONTROL = 5              # PREREG 8.3(2): control arms c1..c5 and m1..m5
N_SURROGATE = 3            # PREREG 8.3.1: phase-randomised eye surrogates
SURROGATE_SEEDS = [97, 98, 99]

HYPOTHESIS_CONDS = ['i_brain', 'ii_eye1', 'ii_eyeall'] + \
                   ['ii_c%d' % k for k in range(1, N_CONTROL + 1)] + \
                   ['ii_m%d' % k for k in range(1, N_CONTROL + 1)] + \
                   ['ii_s%d' % k for k in range(1, N_SURROGATE + 1)]
LADDER_CONDS = ['iii_noica', 'iv_noasr', 'v_wideband']
ALL_CONDS = HYPOTHESIS_CONDS + LADDER_CONDS


# ----------------------------------------------------------------- decomposition
def decompose(path):
    """Run the pipeline once and return everything the conditions need.

    Returns a dict with the preprocessed Raw objects, the fitted ICA, the ICLabel
    probability matrix, and the IC index sets. This is the only expensive step.
    """
    raw = to_raw(load_subject(path))

    # -- ladder rungs iv and v need their own preprocessing chains
    band40 = bandpass(raw)                                  # 0.5-40 Hz
    car40 = car(band40)                                     # + CAR, no ASR   -> iv
    cleaned = asr(car40)                                    # + ASR           -> iii

    wide = raw.copy().notch_filter(NOTCH_HZ, method='fir', phase='zero',
                                   fir_design='firwin')
    wide = bandpass(wide, l_freq=L_FREQ, h_freq=WIDE_HFREQ)
    wide = car(wide)                                        # 0.5-60 + notch  -> v

    # -- single ICA fit, shared by every hypothesis-branch condition
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)
    n_ic = proba.shape[0]

    brain = np.where((proba.argmax(axis=1) == BRAIN) & (proba[:, BRAIN] >= TAU))[0]
    eye = np.where((proba.argmax(axis=1) == EYE) & (proba[:, EYE] >= TAU))[0]
    eye_top = int(eye[proba[eye, EYE].argmax()]) if len(eye) else None
    qualified = set(brain.tolist()) | set(eye.tolist())
    candidates = np.array([j for j in range(n_ic) if j not in qualified], dtype=int)

    return {'raw': raw, 'car40': car40, 'cleaned': cleaned, 'wide': wide,
            'ica': ic, 'proba': proba, 'n_ic': n_ic,
            'brain': brain, 'eye': eye, 'eye_top': eye_top,
            'candidates': candidates}


def _keep(dec, keep_idx):
    """Back-project keeping only keep_idx, dropping every other IC."""
    keep = set(int(i) for i in np.atleast_1d(keep_idx))
    exclude = [j for j in range(dec['n_ic']) if j not in keep]
    return dec['ica'].apply(dec['cleaned'].copy(), exclude=exclude).get_data()


def contributions(dec):
    """Exact sensor-space contribution of each relevant IC, on top of brain-only.

    contribution_j = apply(keep = brain + j) - apply(keep = brain)

    Measured as a back-projection difference rather than || A[:, j] ||^2 var(s_j),
    so MNE's internal PCA and mean handling are carried exactly. Returns
    (base, {j: (19, T) contribution}, {j: total variance}).
    """
    base = _keep(dec, dec['brain'])
    idx = list(dec['candidates'].tolist())
    if dec['eye_top'] is not None:
        idx.append(int(dec['eye_top']))
    contrib, var = {}, {}
    for j in idx:
        c = _keep(dec, np.append(dec['brain'], j)) - base
        contrib[j] = c
        var[j] = float(c.var(axis=1).sum())
    return base, contrib, var


def rank_controls(dec, var):
    """Candidates ordered by closeness of contribution variance to the eye IC."""
    cand = dec['candidates']
    if dec['eye_top'] is None or len(cand) == 0:
        return cand
    target = var[int(dec['eye_top'])]
    return cand[np.argsort([abs(var[int(j)] - target) for j in cand], kind='stable')]


def phase_randomise(x, seed):
    """Phase-randomised surrogate of a (n_ch, T) contribution.

    One random phase sequence is shared by every channel. Because the contribution
    of a single IC is a[:, None] * s[None, :], sharing the phases is exactly
    equivalent to randomising the IC time course itself: the topography and each
    channel's power spectrum are preserved bit for bit, and so is the variance.
    Only the temporal structure -- the blinks -- is destroyed.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    F = np.fft.rfft(x, axis=1)
    phi = rng.uniform(0, 2 * np.pi, size=F.shape[1])
    phi[0] = 0.0                                  # keep DC real
    if n % 2 == 0:
        phi[-1] = 0.0                             # keep Nyquist real
    return np.fft.irfft(F * np.exp(1j * phi)[None, :], n=n, axis=1)


def condition_signals(path, conds=None):
    """{condition_name: (19, T) array} plus a metadata dict.

    Conditions a subject is not eligible for are absent from the result.
    Eligibility: i/ii need at least one brain IC; ii_eye*/ii_s* need a qualified
    eye IC; ii_c*/ii_m* need at least one candidate (arms wrap around beyond that,
    so no arm loses subjects the others keep). Ladder rungs are always available.
    """
    conds = list(ALL_CONDS if conds is None else conds)
    dec = decompose(path)
    brain, eye, eye_top = dec['brain'], dec['eye'], dec['eye_top']
    out = {}
    meta_extra = {'wrapped': 0, 'alpha': [], 'var_eye': np.nan, 'var_c': []}

    if len(brain):
        base, contrib, var = contributions(dec)
        ctrl = rank_controls(dec, var)

        if 'i_brain' in conds:
            out['i_brain'] = base
        if 'ii_eye1' in conds and eye_top is not None:
            out['ii_eye1'] = base + contrib[int(eye_top)]
        if 'ii_eyeall' in conds and len(eye):
            out['ii_eyeall'] = _keep(dec, np.concatenate([brain, eye]))

        if len(ctrl):
            v_eye = var[int(eye_top)] if eye_top is not None else np.nan
            meta_extra['var_eye'] = v_eye
            for k in range(1, N_CONTROL + 1):
                j = int(ctrl[(k - 1) % len(ctrl)])           # wrap-around assignment
                if k > len(ctrl):
                    meta_extra['wrapped'] += 1
                c = contrib[j]
                meta_extra['var_c'].append(var[j])
                if 'ii_c%d' % k in conds:
                    out['ii_c%d' % k] = base + c
                if 'ii_m%d' % k in conds and np.isfinite(v_eye) and var[j] > 0:
                    alpha = float(np.sqrt(v_eye / var[j]))    # match contribution variance
                    meta_extra['alpha'].append(alpha)
                    out['ii_m%d' % k] = base + alpha * c

        if eye_top is not None:
            for k, seed in enumerate(SURROGATE_SEEDS[:N_SURROGATE], start=1):
                if 'ii_s%d' % k in conds:
                    out['ii_s%d' % k] = base + phase_randomise(contrib[int(eye_top)], seed)

    if 'iii_noica' in conds:
        out['iii_noica'] = dec['cleaned'].get_data()
    if 'iv_noasr' in conds:
        out['iv_noasr'] = dec['car40'].get_data()
    if 'v_wideband' in conds:
        out['v_wideband'] = dec['wide'].get_data()

    meta = {'n_ic': dec['n_ic'], 'n_brain': len(brain), 'n_eye': len(eye),
            'n_cand': len(dec['candidates']),
            'eye_prob': (float(dec['proba'][eye_top, EYE]) if eye_top is not None
                         else float(dec['proba'][:, EYE].max())),
            'n_times': dec['cleaned'].n_times}
    meta.update(meta_extra)
    return out, meta, dec


# ------------------------------------------------------------------ diagnostic
def identity_check(dec):
    """ICA is invertible: applying it while excluding nothing must return the input.

    This validates the whole decomposition/back-projection path, which has never
    been checked directly. Returns the max absolute deviation, in the data's own
    units, alongside the signal scale for reference.
    """
    ref = dec['cleaned'].get_data()
    rebuilt = dec['ica'].apply(dec['cleaned'].copy(), exclude=[]).get_data()
    return float(np.abs(rebuilt - ref).max()), float(np.abs(ref).max())


def main(limit):
    subjects = all_subjects()
    if limit:
        subjects = subjects[:limit]

    print('=' * 78)
    print('STAGE 4 DIAGNOSTIC -- reconstruction conditions (PREREG 8.3)')
    print('=' * 78)
    print('band (i-iv)   : %.1f-%.1f Hz' % (L_FREQ, H_FREQ))
    print('band (v)      : %.1f-%.1f Hz + %.0f Hz notch' % (L_FREQ, WIDE_HFREQ, NOTCH_HZ))
    print('tau           : %.2f (argmax and probability >= tau)' % TAU)
    print('control arms  : %d as-is + %d variance-matched + %d phase surrogate'
          % (N_CONTROL, N_CONTROL, N_SURROGATE))
    print('subjects      : %d\n' % len(subjects))

    n_eye_multi = 0
    for i, (path, label) in enumerate(subjects, 1):
        sig, m, dec = condition_signals(path)
        dev, scale = identity_check(dec)
        n_eye_multi += int(m['n_eye'] > 1)

        print('[%d/%d] %-8s %-7s  ICs=%d  brain=%d  eye=%d  candidates=%d  wrapped=%d'
              % (i, len(subjects), path.stem, 'ADHD' if label == 1 else 'Control',
                 m['n_ic'], m['n_brain'], m['n_eye'], m['n_cand'], m['wrapped']))
        print('        eye p=%.2f   identity check max dev %.3g vs scale %.3g'
              % (m['eye_prob'], dev, scale))
        if np.isfinite(m['var_eye']) and m['var_c']:
            print('        contribution variance: eye=%.4g   controls=%s'
                  % (m['var_eye'], ' '.join('%.3g' % v for v in m['var_c'])))
            print('        matching alpha: %s'
                  % ' '.join('%.2f' % a for a in m['alpha']))

        base = sig.get('i_brain')
        bvar = base.var() if base is not None else np.nan
        print('        %-12s %10s %11s' % ('condition', 'shape', 'var ratio'))
        for c in ALL_CONDS:
            if c not in sig:
                print('        %-12s %10s' % (c, 'n/a'))
                continue
            d = sig[c]
            ratio = (d.var() / bvar) if bvar and np.isfinite(bvar) else np.nan
            print('        %-12s %10s %11.3f' % (c, str(d.shape), ratio))
        print()

    print('subjects with more than one qualified eye IC: %d/%d' % (n_eye_multi, len(subjects)))
    print()
    print('Expected reading:')
    print('  - identity deviation many orders below the signal scale (path is exact)')
    print('  - ii_m1..m5 variance ratios should now sit ON TOP of ii_eye1;')
    print('    that is the whole point of the rescaling. ii_c1..c5 stay lower.')
    print('  - ii_s1..s3 must match ii_eye1 to ~3 decimals: same topography, same')
    print('    spectrum, same variance, only the temporal structure differs.')
    print('  - iii/iv/v well above 1 (nothing removed). iii vs iv shows what ASR does.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=3,
                    help='number of subjects to diagnose')
    args = ap.parse_args()
    main(args.limit)