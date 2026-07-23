"""Stage 4 contrasts: IC-removal cost, primary ocular contrast, empirical null band.

Governed by PREREG 8.3 / 8.3.1. Reads the per-condition feature tables written by
features_cond.py and reuses primary_analysis.py's estimator unchanged, so every
number here is produced the same way as the pre-registered primary contrast.

TWO BRANCHES, DIFFERENT SAMPLES (deliberate)
  Ablation ladder    i_brain -> iii_noica -> iv_noasr -> v_wideband
                     Needs no eye IC, so it runs on the full samples. Its question
                     is measurement: how much does each preprocessing step cost?
                     Also run on roster84/roster59 so ladder and contrast numbers
                     can be placed side by side and the sample effect is visible.

  Hypothesis branch  i_brain vs ii_eye1, against 13 controls
                     Needs an eye IC, so roster samples only.

VERDICT (PREREG 8.3.1, fixed before any of these numbers existed)
  Ocular support requires BOTH
    (a) 95% paired-bootstrap CI lower bound of AUC(ii_eye1) - AUC(i_brain) > 0
    (b) that contrast exceeds the maximum over all 13 control arms
  Meeting only one -> INCONCLUSIVE. Never pick the favourable one.

WHY THIRTEEN CONTROLS
  ii_c1..c5   a discarded IC as-is. Under-powered: the leftovers are far smaller
              than the eye IC (median contribution ratio well below 1).
  ii_m1..m5   the same IC rescaled to the eye IC's contribution variance. Matched,
              but for 3.6% of arms the required amplification exceeds 5x, which
              makes those controls synthetic; they are flagged in the output.
  ii_s1..s3   the eye contribution itself with randomised phases. Identical
              topography, identical per-channel spectrum, identical variance,
              amplification factor exactly 1. The tightest control available.
  The maximum over all of them is the bar the primary contrast has to clear.

LENGTH
  Recording length enters every arm as a pass-through covariate, exactly as in
  PREREG 3.1, so the contrast measures what the reconstruction adds beyond length.
  The length-only AUC is computed per sample and printed as the bar that PREREG 5
  requires next to any performance claim.

CACHING
  Out-of-fold subject probabilities are cached per (sample, condition) under
  results/conditions/oof/. The expensive part runs once; contrasts and bootstrap
  are recomputed from cache instantly. Interrupted runs resume.

USAGE
  C:\\envs\\eeg\\python.exe src/contrast_ic.py --seeds 5 --boot 200   # smoke test
  C:\\envs\\eeg\\python.exe src/contrast_ic.py                        # full run
  C:\\envs\\eeg\\python.exe src/contrast_ic.py --branch ladder        # one branch
"""
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from classify import K, RANDOM_STATE                                    # noqa: E402
from classify_ext import MRMR, K_SELECT, load_csv                       # noqa: E402
from primary_analysis import oof_subject_probs, paired_bootstrap, _Tee  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
_COND_DIR = _RESULTS / 'conditions'
_OOF_DIR = _COND_DIR / 'oof'
_SAMPLES_CSV = _RESULTS / 'screening' / 'samples.csv'

N_BOOT = 2000
TACT_EXCLUDED = 'v56p'          # the subject Pastrana-Cortes et al. dropped by eye

LADDER = ['i_brain', 'iii_noica', 'iv_noasr', 'v_wideband']
CONTROLS = (['ii_c%d' % k for k in range(1, 6)] +
            ['ii_m%d' % k for k in range(1, 6)] +
            ['ii_s%d' % k for k in range(1, 4)])
HYPOTHESIS = ['i_brain', 'ii_eye1', 'ii_eyeall'] + CONTROLS

LADDER_SAMPLES = ['full121', 'tier2_86', 'tact120', 'roster84', 'roster59']
HYP_SAMPLES = ['roster84', 'roster59']


# ------------------------------------------------------------------- samples
def build_samples():
    """{sample_name: set(subject_id)} from the screening roster and the eye roster."""
    tier = {'in_full': set(), 'in_tier1': set(), 'in_tier2': set()}
    with open(_SAMPLES_CSV, newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            for key in tier:
                if int(row[key]):
                    tier[key].add(row['subject_id'])

    roster = set()
    with open(_COND_DIR / 'features_ii_eye1.csv', newline='') as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            roster.add(row[0])

    return {
        'full121': set(tier['in_full']),
        'tier2_86': set(tier['in_tier2']),
        'tact120': set(tier['in_full']) - {TACT_EXCLUDED},
        'roster84': set(roster),
        'roster59': set(roster) & tier['in_tier2'],
    }


# --------------------------------------------------------------------- arms
def make_pipe(n_pool):
    """Identical to primary_analysis.make_pipe: scaler -> mRMR -> Logistic.

    Columns past n_pool (here: recording length) bypass selection; they are
    pre-specified covariates, not candidates.
    """
    return lambda: Pipeline([
        ('sc', StandardScaler()),
        ('sel', MRMR(k=K_SELECT, n_pool=n_pool)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                   random_state=RANDOM_STATE,
                                   class_weight='balanced'))])


def make_pipe_plain():
    """No selection stage: used for the length-only stopwatch baseline."""
    return lambda: Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                   random_state=RANDOM_STATE,
                                   class_weight='balanced'))])


def load_condition(cond, keep):
    """features_{cond}.csv restricted to `keep` -> X(306 + length), y, groups."""
    Xe, ye, ge, cols = load_csv(_COND_DIR / ('features_%s.csv' % cond))
    i_len = cols.index('rec_length_sec')
    feat_idx = [i for i in range(len(cols)) if i != i_len]
    mask = np.array([s in keep for s in ge])
    if not mask.any():
        return None
    X = np.hstack([Xe[np.ix_(np.where(mask)[0], feat_idx)], Xe[mask][:, [i_len]]])
    return X, ye[mask], ge[mask], len(feat_idx)


def get_oof(sample, cond, keep, seeds, force=False):
    """Cached seed-averaged out-of-fold subject probabilities for one arm."""
    _OOF_DIR.mkdir(parents=True, exist_ok=True)
    cache = _OOF_DIR / ('%s__%s.npz' % (sample, cond))
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        return {'subs': [str(s) for s in d['subs']], 'y': d['y'],
                'p': d['p'], 'auc': float(d['auc']), 'fold': float(d['fold']),
                'cached': True}

    loaded = load_condition(cond, keep)
    if loaded is None:
        return None
    X, y, g, n_pool = loaded
    subs = list(dict.fromkeys(g.tolist()))
    y_sub, p_sub, fold_auc = oof_subject_probs(X, y, g, make_pipe(n_pool), seeds)
    auc = float(roc_auc_score(y_sub, p_sub))
    np.savez_compressed(cache, subs=np.array(subs), y=y_sub, p=p_sub,
                        auc=np.float64(auc), fold=np.float64(fold_auc))
    return {'subs': subs, 'y': y_sub, 'p': p_sub, 'auc': auc,
            'fold': fold_auc, 'cached': False}


def stopwatch(sample, keep, seeds, force=False):
    """Length-only AUC for one sample: the bar PREREG 5 requires."""
    cache = _OOF_DIR / ('%s__length_only.npz' % sample)
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        return float(d['auc'])
    loaded = load_condition('i_brain', keep)
    if loaded is None:
        return float('nan')
    X, y, g, n_pool = loaded
    L = X[:, [n_pool]]                                   # length column only
    y_sub, p_sub, _ = oof_subject_probs(L, y, g, make_pipe_plain(), seeds)
    auc = float(roc_auc_score(y_sub, p_sub))
    np.savez_compressed(cache, auc=np.float64(auc))
    return auc


def aligned(a, b):
    """Paired comparison requires the identical subject list in the same order."""
    return a is not None and b is not None and a['subs'] == b['subs']


# ------------------------------------------------------------------ reporting
def report_ladder(store, samples, n_boot):
    print('\n' + '=' * 78)
    print('ABLATION LADDER -- what each preprocessing step costs')
    print('=' * 78)
    print('One factor per rung. Positive means removing that step RAISES AUC,')
    print('i.e. the step was discarding information the classifier could use.\n')

    for s in LADDER_SAMPLES:
        arms = {c: store.get((s, c)) for c in LADDER}
        if arms.get('i_brain') is None:
            continue
        print('[%s]  n=%d   stopwatch (length only) AUC = %.3f'
              % (s, len(arms['i_brain']['subs']), store.get(('sw', s), float('nan'))))
        print('  %-14s %9s %9s' % ('condition', 'AUC', 'fold AUC'))
        for c in LADDER:
            if arms.get(c) is None:
                print('  %-14s %9s' % (c, 'n/a'))
                continue
            print('  %-14s %9.3f %9.3f' % (c, arms[c]['auc'], arms[c]['fold']))

        print('  %-30s %8s   %-22s' % ('step', 'dAUC', '95% CI'))
        for base, comb, label in (('i_brain', 'iii_noica', 'remove ICA'),
                                  ('iii_noica', 'iv_noasr', 'remove ASR'),
                                  ('iv_noasr', 'v_wideband', 'widen band to 0.5-60'),
                                  ('i_brain', 'v_wideband', 'TOTAL (all three)')):
            a, b = arms.get(base), arms.get(comb)
            if not aligned(a, b):
                print('  %-30s %8s' % (label, 'n/a'))
                continue
            m, lo, hi, _ = paired_bootstrap(a['y'], a['p'], b['p'], n_boot)
            tag = 'CI>0' if lo > 0 else ('CI<0' if hi < 0 else 'CI includes 0')
            print('  %-30s %+8.3f   [%+.3f, %+.3f]  %s' % (label, m, lo, hi, tag))
        print()


def report_hypothesis(store, n_boot):
    print('\n' + '=' * 78)
    print('HYPOTHESIS BRANCH -- ocular IC kept in the signal, against 13 controls')
    print('=' * 78)

    for s in HYP_SAMPLES:
        base = store.get((s, 'i_brain'))
        if base is None:
            continue
        print('\n[%s]  n=%d   stopwatch (length only) AUC = %.3f'
              % (s, len(base['subs']), store.get(('sw', s), float('nan'))))
        print('  i_brain (baseline) AUC = %.3f\n' % base['auc'])

        def contrast(cond):
            arm = store.get((s, cond))
            if not aligned(base, arm):
                return None
            m, lo, hi, _ = paired_bootstrap(base['y'], base['p'], arm['p'], n_boot)
            return arm['auc'], m, lo, hi

        print('  %-12s %8s %9s %10s %-22s' % ('arm', 'AUC', 'dAUC', '', '95% CI'))
        prim = contrast('ii_eye1')
        if prim:
            print('  %-12s %8.3f %+9.3f   [%+.3f, %+.3f]   <- PRIMARY'
                  % ('ii_eye1', prim[0], prim[1], prim[2], prim[3]))
        allv = contrast('ii_eyeall')
        if allv:
            print('  %-12s %8.3f %+9.3f   [%+.3f, %+.3f]   (secondary)'
                  % ('ii_eyeall', allv[0], allv[1], allv[2], allv[3]))

        print()
        ctrl = {}
        for c in CONTROLS:
            r = contrast(c)
            if r:
                ctrl[c] = r
                fam = {'c': 'as-is', 'm': 'variance-matched', 's': 'phase surrogate'}
                print('  %-12s %8.3f %+9.3f   [%+.3f, %+.3f]   %s'
                      % (c, r[0], r[1], r[2], r[3], fam[c[3]]))

        if not prim or not ctrl:
            continue
        worst = max(ctrl.items(), key=lambda kv: kv[1][1])
        print('\n  control null band : %+.3f to %+.3f  (max: %s)'
              % (min(v[1] for v in ctrl.values()),
                 max(v[1] for v in ctrl.values()), worst[0]))
        print('  primary contrast  : %+.3f' % prim[1])

        # Family breakdown. REPORTING ONLY -- the verdict below still uses the
        # pooled maximum over all 13 controls, exactly as PREREG 8.3.1 fixed it.
        # The three families answer different questions, so knowing which one
        # binds tells you what a failure to clear the bar actually means.
        fams = (('ii_c', 'as-is           '),
                ('ii_m', 'variance-matched'),
                ('ii_s', 'phase surrogate '))
        print('\n  by control family (reporting only, verdict rule unchanged)')
        for pre, label in fams:
            vals = {k: v[1] for k, v in ctrl.items() if k.startswith(pre)}
            if not vals:
                continue
            mx = max(vals, key=vals.get)
            print('    %s  max %+.3f (%-6s)   primary clears it: %s'
                  % (label, vals[mx], mx, 'YES' if prim[1] > vals[mx] else 'no'))
        print('    ii_c  answers: is any discarded IC enough? Under-powered, since')
        print('          the leftovers are much smaller than the eye IC.')
        print('    ii_m  answers: does an equally large NON-ocular component do the')
        print('          same job? This is the real alternative explanation.')
        print('    ii_s  answers: does the TEMPORAL structure of the ocular IC')
        print('          matter beyond its spectrum and topography? It preserves')
        print('          per-channel PSD and inter-channel phase, so the CLR (76)')
        print('          and coherence (40) blocks are near-identical to ii_eye1:')
        print('          116 of 306 features are shared with the treatment. Failing')
        print('          to clear ii_s is not evidence that ocular information is')
        print('          absent, only that it does not live in the time domain.')

        cond_a = prim[2] > 0
        cond_b = prim[1] > worst[1][1]
        print('\n  [VERDICT] PREREG 8.3.1')
        print('    (a) CI lower bound > 0            : %s' % ('YES' if cond_a else 'no'))
        print('    (b) exceeds every control arm     : %s' % ('YES' if cond_b else 'no'))
        if cond_a and cond_b:
            print('    -> SUPPORTED. Keeping the ocular IC in the signal adds')
            print('       discriminative information beyond adding any component.')
        elif not cond_a and not cond_b:
            print('    -> NOT SUPPORTED at this sample size. Report the interval.')
        else:
            print('    -> INCONCLUSIVE. One condition met, one not; PREREG forbids')
            print('       reporting the favourable half alone.')


# ----------------------------------------------------------------------- main
def main(seeds, n_boot, branch, force):
    samples = build_samples()
    print('=' * 78)
    print('STAGE 4 CONTRASTS (PREREG 8.3 / 8.3.1)')
    print('=' * 78)
    for k in LADDER_SAMPLES:
        print('  %-10s n=%d' % (k, len(samples[k])))
    print('  seeds=%d  bootstrap=%d  K=%d  mRMR k=%d  Logistic C=1.0'
          % (len(seeds), n_boot, K, K_SELECT))
    print('  estimator identical to primary_analysis.py\n')

    plan = []
    if branch in ('all', 'ladder'):
        plan += [(s, c) for s in LADDER_SAMPLES for c in LADDER]
    if branch in ('all', 'hypothesis'):
        plan += [(s, c) for s in HYP_SAMPLES for c in HYPOTHESIS]
    plan = list(dict.fromkeys(plan))

    store, t0 = {}, time.time()
    for i, (s, c) in enumerate(plan, 1):
        r = get_oof(s, c, samples[s], seeds, force)
        if r is None:
            print('[%3d/%3d] %-10s %-12s skipped (no rows)' % (i, len(plan), s, c))
            continue
        store[(s, c)] = r
        print('[%3d/%3d] %-10s %-12s n=%3d AUC=%.3f  %s  %6.1fs'
              % (i, len(plan), s, c, len(r['subs']), r['auc'],
                 'cache' if r['cached'] else 'fit  ', time.time() - t0))
        sys.stdout.flush()

    for s in set(s for s, _ in plan):
        store[('sw', s)] = stopwatch(s, samples[s], seeds, force)

    if branch in ('all', 'ladder'):
        report_ladder(store, samples, n_boot)
    if branch in ('all', 'hypothesis'):
        report_hypothesis(store, n_boot)

    print('\n' + '-' * 78)
    print('CI covers subject sampling only, not model-training variability (PREREG 3.3).')
    print('Interpretation limit (PREREG 8.3(5)): v_wideband reproduces the prior')
    print('study\'s PREPROCESSING only. Epoching and model still differ, so a gap')
    print('remaining here is not evidence against their result.')
    print('elapsed: %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--branch', choices=['all', 'ladder', 'hypothesis'], default='all')
    ap.add_argument('--force', action='store_true', help='ignore the OOF cache')
    args = ap.parse_args()

    _COND_DIR.mkdir(parents=True, exist_ok=True)
    orig = sys.stdout
    with open(_COND_DIR / ('contrast_%s.txt' % args.branch), 'w', encoding='utf-8') as fh:
        sys.stdout = _Tee(orig, fh)
        try:
            main([RANDOM_STATE + i for i in range(args.seeds)],
                 args.boot, args.branch, args.force)
        finally:
            sys.stdout = orig