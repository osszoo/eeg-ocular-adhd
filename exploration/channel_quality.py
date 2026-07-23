"""Objective channel-quality screening (branch: benchmark, step 1).

WHY THIS EXISTS
  The pipeline has never had a bad-channel detection step. CAR subtracts the
  19-channel mean at every sample, so a noisy channel is injected into all 18
  others at 1/19 strength, and ASR, ICA and ICLabel all see that contaminated
  signal. A dead channel harms through a different path -- its 16 channel-level
  features in features_ext.py are garbage and it distorts the region means --
  but the conclusion is the same: one bad channel is enough to corrupt that
  subject's whole pipeline. A prior study on this dataset (Pastrana-Cortes
  et al., Technologies 2026) dropped subject v56p by visual inspection; this
  script replaces visual inspection with a published objective standard.

WHERE IT MEASURES
  On the loaded Raw, BEFORE bandpass and BEFORE CAR.
    - Before CAR is mandatory. CAR smears contamination over all channels and
      pulls the offending channel toward the mean, which destroys detectability.
      PREP detects bad channels before referencing for exactly this reason.
    - Before bandpass because the high-frequency-noise criterion inspects
      >50 Hz content, which the 40 Hz lowpass would erase. At 128 Hz the
      50-64 Hz band exists and carries the 50 Hz mains, making that criterion a
      sensitive proxy for electrode contact quality. Note this band lies outside
      the 0.5-40 Hz analysis band, so hf_noise is an indicator, not the harm.
  Low-frequency drift is handled by pyprep's own internal 1 Hz detrend.

HOW IT DECIDES (fixed in PREREG 8.1 before this script was first run)
  pyprep 0.7.1 NoisyChannels with the published PREP defaults. No threshold is
  chosen by this study. RANSAC is off: with 19 channels a 25% predictor subset
  is 4-5 channels, so spherical-spline prediction is meaningless, and it would
  add a random seed as a new degree of freedom.
    tier1 = nan | flat | dropout | correlation | hf_noise
    tier2 = tier1 | deviation                          <- PRIMARY sample
  A subject holding >= 1 bad channel under a tier is excluded from that tier.
  No interpolation: an interpolated channel is a linear combination of the
  others, which lowers rank -> changes ICA n_components -> changes the IC
  decomposition -> changes ICLabel verdicts -> changes the 84-subject roster.
  All three samples (full / tier1 / tier2) are always reported together.

READ-ONLY
  This script does not modify the pipeline, does not run ICA, and does not
  invalidate the brain-epoch cache. Its only products are a log and a roster.

OUTPUT
  results/screening/channel_quality.txt   full log (mirror of stdout)
  results/screening/samples.csv           per-subject roster for later steps

USAGE
  C:\\envs\\eeg\\python.exe exploration/channel_quality.py
  C:\\envs\\eeg\\python.exe exploration/channel_quality.py --limit 5
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
from load_data import all_subjects, load_subject, to_raw, SFREQ, CH   # noqa: E402

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
_OUT_DIR = _RESULTS / 'screening'

# Frontal channels carry the ocular signal this study hypothesises about.
# Their flag rate per criterion is reported so that construct validity can be
# judged; it never feeds back into the rule (PREREG 8.1 item 4).
FRONTAL = ['Fp1', 'Fp2', 'F7', 'F8']

CRITERIA = ['nan', 'flat', 'dropout', 'correlation', 'hf_noise', 'deviation']
TIER1 = ['nan', 'flat', 'dropout', 'correlation', 'hf_noise']
TIER2 = TIER1 + ['deviation']

RANDOM_STATE = 97      # matches the ICA seed convention; unused while ransac=False


# ------------------------------------------------------------------ screening
def screen_subject(path):
    """Run PREP noisy-channel detection on one subject.

    Returns (flags, n_samples) where flags maps criterion -> list of channel
    names. nan/flat is computed by NoisyChannels.__init__ already; it is called
    again explicitly so the criterion order is readable in this file.
    """
    from pyprep import NoisyChannels

    arr = load_subject(path)
    raw = to_raw(arr)                       # loaded Raw only: no bandpass, no CAR
    nc = NoisyChannels(raw, do_detrend=True, random_state=RANDOM_STATE,
                       ransac=False)
    nc.find_bad_by_nan_flat()               # must precede the others
    nc.find_bad_by_correlation()            # also fills bad_by_dropout
    nc.find_bad_by_hfnoise()
    nc.find_bad_by_deviation()

    flags = {
        'nan':         list(nc.bad_by_nan),
        'flat':        list(nc.bad_by_flat),
        'dropout':     list(nc.bad_by_dropout),
        'correlation': list(nc.bad_by_correlation),
        'hf_noise':    list(nc.bad_by_hf_noise),
        'deviation':   list(nc.bad_by_deviation),
    }
    return flags, arr.shape[0]


def union(flags, criteria):
    """Channels flagged by at least one of the given criteria, in montage order."""
    bad = set()
    for c in criteria:
        bad.update(flags[c])
    return [ch for ch in CH if ch in bad]


# ------------------------------------------------------------------ reporting
def fmt_flags(flags):
    """One-line summary of which criteria fired for a subject."""
    parts = []
    for c in CRITERIA:
        if flags[c]:
            parts.append('%s:%s' % (c[:4], ','.join(flags[c])))
    return ' | '.join(parts) if parts else 'clean'


def median_or_nan(values):
    return float(np.median(values)) if values else float('nan')


def sample_report(rows, criteria, title):
    """Print counts, group balance and length distribution for one tier."""
    kept = [r for r in rows if not union(r['flags'], criteria)] if criteria else list(rows)
    drop = [r for r in rows if r not in kept]

    n_a = sum(r['label'] == 1 for r in kept)
    n_c = sum(r['label'] == 0 for r in kept)
    tot_a = sum(r['label'] == 1 for r in rows)
    tot_c = sum(r['label'] == 0 for r in rows)
    ex_a = tot_a - n_a
    ex_c = tot_c - n_c
    pa = ex_a / tot_a if tot_a else 0.0
    pc = ex_c / tot_c if tot_c else 0.0

    la = [r['sec'] for r in kept if r['label'] == 1]
    lc = [r['sec'] for r in kept if r['label'] == 0]
    ma, mc = median_or_nan(la), median_or_nan(lc)

    print('  %-6s  keep %3d (ADHD %2d / Control %2d)   exclude %3d' %
          (title, len(kept), n_a, n_c, len(drop)))
    print('          exclusion rate  ADHD %4.1f%%  Control %4.1f%%  gap %4.1f%%p' %
          (100 * pa, 100 * pc, 100 * abs(pa - pc)))
    print('          length median   ADHD %6.1fs  Control %6.1fs  delta %+6.1fs' %
          (ma, mc, ma - mc))
    return kept, drop


def main(limit=None):
    subjects = all_subjects()
    if limit:
        subjects = subjects[:limit]

    print('=' * 78)
    print('CHANNEL QUALITY SCREENING  --  pyprep NoisyChannels, PREP defaults')
    print('=' * 78)
    try:
        import pyprep
        print('pyprep version : %s' % pyprep.__version__)
    except ImportError:
        print('pyprep is not installed.  C:\\envs\\eeg\\python.exe -m pip install pyprep')
        return
    print('measured on    : loaded Raw, before bandpass and before CAR')
    print('detrend        : pyprep internal 1 Hz highpass')
    print('thresholds     : flat 1e-15 | corr 0.4 / 1s / 1% | hf z>5 | dev z>5')
    print('ransac         : OFF (19 channels: 25% subset is 4-5 predictors)')
    print('tier1          : %s' % ' U '.join(TIER1))
    print('tier2 (PRIMARY): tier1 U deviation')
    print('subjects       : %d' % len(subjects))
    print()

    rows = []
    t0 = time.time()
    print('--- per subject ' + '-' * 61)
    for i, (path, label) in enumerate(subjects, 1):
        flags, n_samples = screen_subject(path)
        sec = n_samples / SFREQ
        rows.append({'name': path.stem, 'label': int(label), 'sec': sec,
                     'flags': flags})
        print('  [%3d/%3d] %-8s %-7s %6.1fs  %s' %
              (i, len(subjects), path.stem,
               'ADHD' if label == 1 else 'Control', sec, fmt_flags(flags)))
        sys.stdout.flush()
    print('  elapsed %.1f min' % ((time.time() - t0) / 60.0))
    print()

    # -- criterion x channel matrix: is any criterion eating frontal channels?
    print('--- flag counts: channel x criterion ' + '-' * 40)
    print('  %-5s %s' % ('ch', ''.join('%12s' % c for c in CRITERIA)))
    for ch in CH:
        counts = [sum(ch in r['flags'][c] for r in rows) for c in CRITERIA]
        mark = ' <- frontal' if ch in FRONTAL else ''
        print('  %-5s %s%s' % (ch, ''.join('%12d' % n for n in counts), mark))
    print()

    print('--- frontal vs non-frontal flag rate (per channel-subject) ' + '-' * 19)
    n_sub = len(rows)
    print('  %-12s %10s %12s %8s' % ('criterion', 'frontal', 'non-frontal', 'ratio'))
    for c in CRITERIA:
        f_hits = sum(sum(ch in r['flags'][c] for ch in FRONTAL) for r in rows)
        o_hits = sum(sum(ch in r['flags'][c] for ch in CH if ch not in FRONTAL)
                     for r in rows)
        f_rate = f_hits / float(n_sub * len(FRONTAL))
        o_rate = o_hits / float(n_sub * (len(CH) - len(FRONTAL)))
        ratio = (f_rate / o_rate) if o_rate > 0 else float('inf')
        print('  %-12s %9.2f%% %11.2f%% %8s' %
              (c, 100 * f_rate, 100 * o_rate,
               '%.2f' % ratio if np.isfinite(ratio) else 'inf'))
    print('  (reported for construct validity only; the rule is not changed by it)')
    print()

    # -- bad-channel count distribution
    print('--- bad channels per subject ' + '-' * 48)
    for name, crit in (('tier1', TIER1), ('tier2', TIER2)):
        dist = {}
        for r in rows:
            k = len(union(r['flags'], crit))
            dist[k] = dist.get(k, 0) + 1
        line = '  '.join('%d ch: %d' % (k, dist[k]) for k in sorted(dist))
        print('  %-6s %s' % (name, line))
    print()

    # -- the three samples
    print('--- samples ' + '-' * 65)
    sample_report(rows, [], 'full')
    keep1, _ = sample_report(rows, TIER1, 'tier1')
    keep2, _ = sample_report(rows, TIER2, 'tier2')
    print()

    # -- intersection with the 84-subject ocular roster (primary analysis sample)
    roster = ocular_roster()
    if roster is None:
        print('--- ocular roster: results/X_ocular84.csv not found, skipped')
    else:
        print('--- intersection with the ocular-IC roster (%d subjects) %s'
              % (len(roster), '-' * 20))
        for name, kept in (('full', rows), ('tier1', keep1), ('tier2', keep2)):
            inter = [r for r in kept if r['name'] in roster]
            n_a = sum(r['label'] == 1 for r in inter)
            n_c = sum(r['label'] == 0 for r in inter)
            print('  %-6s %3d subjects (ADHD %2d / Control %2d)'
                  % (name, len(inter), n_a, n_c))
        print('  -> this is the sample the pre-registered primary contrast runs on')
    print()

    # -- external cross-check: the subject a prior study dropped by eye
    print('--- external cross-check: v56p ' + '-' * 46)
    hit = [r for r in rows if r['name'] == 'v56p']
    if not hit:
        print('  v56p not present in this run')
    else:
        r = hit[0]
        print('  v56p  %s' % fmt_flags(r['flags']))
        print('  tier1 bad: %s' % (union(r['flags'], TIER1) or 'none'))
        print('  tier2 bad: %s' % (union(r['flags'], TIER2) or 'none'))
        print('  (Pastrana-Cortes et al. dropped this subject for FP1/FP2 contamination)')
    print()

    write_csv(rows)
    print('--- written ' + '-' * 65)
    print('  %s' % (_OUT_DIR / 'channel_quality.txt'))
    print('  %s' % (_OUT_DIR / 'samples.csv'))


def ocular_roster():
    """Subject ids that hold a qualifying eye IC, read from the assembled table."""
    path = _RESULTS / 'X_ocular84.csv'
    if not path.exists():
        return None
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        next(r)
        return {row[0] for row in r}


def write_csv(rows):
    """Machine-readable roster consumed by later steps (epoch overlap, models)."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUT_DIR / 'samples.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'label', 'rec_sec',
                    'n_bad_tier1', 'n_bad_tier2', 'bad_tier1', 'bad_tier2',
                    'in_full', 'in_tier1', 'in_tier2'] +
                   ['bad_' + c for c in CRITERIA])
        for r in rows:
            b1 = union(r['flags'], TIER1)
            b2 = union(r['flags'], TIER2)
            w.writerow([r['name'], r['label'], '%.3f' % r['sec'],
                        len(b1), len(b2), ';'.join(b1), ';'.join(b2),
                        1, int(not b1), int(not b2)] +
                       [';'.join(r['flags'][c]) for c in CRITERIA])


class _Tee:
    """Console and file at the same time."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='screen only the first N subjects (smoke test)')
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    with open(_OUT_DIR / 'channel_quality.txt', 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            main(limit=args.limit)
        finally:
            sys.stdout = orig