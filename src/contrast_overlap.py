"""Stage 2 contrast: does 50% epoch overlap help? (PREREG 8.3(6), 8.3.3(3))

Exploratory arm. Reuses contrast_ic.py's machinery unchanged, so the estimator
is identical to the pre-registered primary analysis: Logistic C=1.0, mRMR k=30
inside folds, StratifiedGroupKFold K=5, 50 seeds, pooled out-of-fold AUC,
subject-level paired bootstrap.

THE CONTRAST
  ov50 - ov00, both cut from the SAME reassembled brain-only signal, so the only
  thing that differs is the stride. Recording length rides along in both arms as
  a pass-through covariate, as in PREREG 3.1.

PROMOTION RULE (fixed in PREREG 8.3(6) before any of this existed)
  Overlap is promoted to the primary design only if dAUC >= +0.03 on ALL THREE
  tier samples: full121, tier1_91, tier2_86. The roster samples and tact120 are
  reported but do not participate in the decision. Judging on this neural-only
  arm keeps the ocular contrast unseen at decision time (same structure as the
  PREREG 3.7 blinding).

BUILT-IN VALIDATION
  ov00 is a rebuild of the stage 4 i_brain condition from cached epochs. Its AUC
  is printed next to the stage 4 value. Agreement to about 0.001 means the
  reassembled signal is the signal stage 4 used, which is the check the
  feature-level deviation in features_overlap.py could not settle: that one
  compared absolute differences across 306 features of wildly different scale.

WHAT A NULL HERE DOES NOT MEAN
  The prior study used 2-second epochs at 50% overlap. Only overlap is testable
  here; 2 seconds would leave region coherence with a single Welch window and
  require redefining features, which PREREG 3.6 forbids. So dAUC near zero means
  "overlap alone does not close the gap", not "epoching is not the cause".

USAGE
  C:\\envs\\eeg\\python.exe src/contrast_overlap.py --seeds 5 --boot 200
  C:\\envs\\eeg\\python.exe src/contrast_overlap.py
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                                    # noqa: E402
from classify_ext import K_SELECT, load_csv                             # noqa: E402
from primary_analysis import oof_subject_probs, paired_bootstrap, _Tee  # noqa: E402
from contrast_ic import (build_samples, make_pipe, make_pipe_plain,     # noqa: E402
                         metrics, LADDER_SAMPLES)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
_OV_DIR = _RESULTS / 'overlap'
_OOF_DIR = _OV_DIR / 'oof'

CONDS = ['ov00', 'ov50']
DECISION_SAMPLES = ['full121', 'tier1_91', 'tier2_86']    # PREREG 8.3(6)
PROMOTION_THRESHOLD = 0.03
STAGE4_REFERENCE = {'full121': 0.798, 'tier1_91': 0.807, 'tier2_86': 0.785,
                    'tact120': 0.794, 'roster84': 0.713, 'roster59': 0.720}


def load_overlap(cond, keep):
    """features_i_brain_{cond}.csv restricted to `keep` -> X(306 + length), y, g."""
    Xe, ye, ge, cols = load_csv(_OV_DIR / ('features_i_brain_%s.csv' % cond))
    i_len = cols.index('rec_length_sec')
    feat_idx = [i for i in range(len(cols)) if i != i_len]
    mask = np.array([s in keep for s in ge])
    if not mask.any():
        return None
    X = np.hstack([Xe[np.ix_(np.where(mask)[0], feat_idx)], Xe[mask][:, [i_len]]])
    return X, ye[mask], ge[mask], len(feat_idx)


def get_oof(sample, cond, keep, seeds, force=False):
    _OOF_DIR.mkdir(parents=True, exist_ok=True)
    cache = _OOF_DIR / ('%s__%s.npz' % (sample, cond))
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        return {'subs': [str(s) for s in d['subs']], 'y': d['y'], 'p': d['p'],
                'auc': float(d['auc']), 'fold': float(d['fold']),
                'nep': int(d['nep']), 'cached': True}
    loaded = load_overlap(cond, keep)
    if loaded is None:
        return None
    X, y, g, n_pool = loaded
    subs = list(dict.fromkeys(g.tolist()))
    y_sub, p_sub, fold_auc = oof_subject_probs(X, y, g, make_pipe(n_pool), seeds)
    auc = float(roc_auc_score(y_sub, p_sub))
    np.savez_compressed(cache, subs=np.array(subs), y=y_sub, p=p_sub,
                        auc=np.float64(auc), fold=np.float64(fold_auc),
                        nep=np.int64(len(y)))
    return {'subs': subs, 'y': y_sub, 'p': p_sub, 'auc': auc, 'fold': fold_auc,
            'nep': len(y), 'cached': False}


def stopwatch(sample, keep, seeds, force=False):
    cache = _OOF_DIR / ('%s__length_only.npz' % sample)
    if cache.exists() and not force:
        return float(np.load(cache, allow_pickle=False)['auc'])
    loaded = load_overlap('ov00', keep)
    if loaded is None:
        return float('nan')
    X, y, g, n_pool = loaded
    y_sub, p_sub, _ = oof_subject_probs(X[:, [n_pool]], y, g, make_pipe_plain(), seeds)
    auc = float(roc_auc_score(y_sub, p_sub))
    np.savez_compressed(cache, auc=np.float64(auc))
    return auc


def main(seeds, n_boot, force):
    samples = build_samples()
    print('=' * 78)
    print('STAGE 2 CONTRAST -- 50%% epoch overlap (PREREG 8.3(6))')
    print('=' * 78)
    print('  seeds=%d  bootstrap=%d  K=%d  mRMR k=%d  Logistic C=1.0'
          % (len(seeds), n_boot, K, K_SELECT))
    print('  promotion needs dAUC >= %+.2f on ALL of: %s'
          % (PROMOTION_THRESHOLD, ', '.join(DECISION_SAMPLES)))
    print('  signal fixed to brain-only (PREREG 8.3.3(3)); no stacking of stage 4\n')

    store, t0 = {}, time.time()
    for s in LADDER_SAMPLES:
        for c in CONDS:
            r = get_oof(s, c, samples[s], seeds, force)
            if r is None:
                continue
            store[(s, c)] = r
            print('  %-10s %-5s n=%3d epochs=%5d AUC=%.3f  %s  %6.1fs'
                  % (s, c, len(r['subs']), r['nep'], r['auc'],
                     'cache' if r['cached'] else 'fit  ', time.time() - t0))
            sys.stdout.flush()
        store[('sw', s)] = stopwatch(s, samples[s], seeds, force)

    print('\n' + '=' * 78)
    print('VALIDATION -- ov00 must reproduce the stage 4 i_brain condition')
    print('=' * 78)
    print('  %-10s %10s %10s %8s' % ('sample', 'ov00 AUC', 'stage4', 'diff'))
    for s in LADDER_SAMPLES:
        a = store.get((s, 'ov00'))
        if a is None:
            continue
        ref = STAGE4_REFERENCE.get(s, float('nan'))
        print('  %-10s %10.3f %10.3f %+8.3f' % (s, a['auc'], ref, a['auc'] - ref))
    print('  Differences at the third decimal mean the reassembled signal is the')
    print('  one stage 4 used, and the feature-level deviation was rounding.')

    print('\n' + '=' * 78)
    print('CONTRAST -- ov50 minus ov00')
    print('=' * 78)
    decision = {}
    for s in LADDER_SAMPLES:
        a, b = store.get((s, 'ov00')), store.get((s, 'ov50'))
        if a is None or b is None or a['subs'] != b['subs']:
            print('  %-10s n/a' % s)
            continue
        mark = '  <- decision sample' if s in DECISION_SAMPLES else ''
        print('\n[%s]  n=%d   stopwatch AUC = %.3f%s'
              % (s, len(a['subs']), store.get(('sw', s), float('nan')), mark))
        print('  %-6s %7s %7s %7s %7s %8s' % ('arm', 'AUC', 'acc', 'sens', 'spec',
                                              'epochs'))
        for c, r in (('ov00', a), ('ov50', b)):
            acc, sens, spec = metrics(r['y'], r['p'])
            print('  %-6s %7.3f %7.3f %7.3f %7.3f %8d'
                  % (c, r['auc'], acc, sens, spec, r['nep']))
        m, lo, hi, _ = paired_bootstrap(a['y'], a['p'], b['p'], n_boot)
        tag = 'CI>0' if lo > 0 else ('CI<0' if hi < 0 else 'CI includes 0')
        print('  dAUC %+.3f   [%+.3f, %+.3f]   %s' % (m, lo, hi, tag))
        decision[s] = m

    print('\n' + '=' * 78)
    print('[PROMOTION VERDICT] PREREG 8.3(6)')
    print('=' * 78)
    ok = True
    for s in DECISION_SAMPLES:
        v = decision.get(s)
        if v is None:
            print('  %-10s missing -> cannot promote' % s)
            ok = False
            continue
        passed = v >= PROMOTION_THRESHOLD
        ok = ok and passed
        print('  %-10s dAUC %+.3f  >= %+.2f : %s'
              % (s, v, PROMOTION_THRESHOLD, 'YES' if passed else 'no'))
    if ok:
        print('\n  -> PROMOTE. Record the change as a separate deviation entry and')
        print('     rebuild the primary design on overlapping epochs.')
    else:
        print('\n  -> DO NOT PROMOTE. Overlap stays an exploratory arm; the primary')
        print('     design remains 6 s non-overlapping epochs.')
        print('     This does not clear epoching as a cause of the gap to the prior')
        print('     study: only overlap was testable, not the 2 s epoch length.')

    print('\nelapsed: %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    _OV_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    with open(_OV_DIR / 'contrast_overlap.txt', 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            main([RANDOM_STATE + i for i in range(args.seeds)], args.boot, args.force)
        finally:
            sys.stdout = orig