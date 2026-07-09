"""특징추출 파이프라인(정식 출처).

전처리(preprocess.py)까지 끝난 신호에서 분류용 특징을 뽑아 (에폭 × 특징) 표로
만든다. 진단 스크립트(exploration/)가 검증한 로직을 여기서 정식 함수로 승격한다 —
앞으로 특징 계산의 single source of truth는 이 파일이다.

설계(확정):
  - 에폭 : 6초, 겹침 없음 (epoch_estimator_sweep로 확정)
  - 추정기 : 단일 FFT(주기도표) (동 진단으로 확정)
  - 표 단위 : 한 행 = 6초 에폭 하나. 한 피험자는 여러 행.
              CV는 subject_id로 묶어 분할(에폭이 train/test에 안 섞이게).

담는 것:
  - 안구 power류 = 저주파 파워비율(0.5–4 / 0.5–40Hz), top-1 안구 IC(집계 B), 84명.
  - 뇌파 = brain IC만 back-projection(집계 C)해 19채널 신경신호 복원 →
           부위별 상대 밴드파워(δθαβ, 분모=4대역합=0.5–30Hz) 20개 + TBR 5개 = 25개, 121명.

캐시·산출물:
  - 피험자별 캐시(기계장치) : exploration/_cache/feat_{ocular,neural}_{name}.npz
  - 최종 표(산출물)          : results/features_{ocular,neural}.csv

누수 경계 : 특징 계산은 그룹 라벨을 보지 않는다. IC 판정도 라벨이 아니라 ICLabel
       기준(argmax & ≥0.5)으로만 한다. z-정규화는 train fold에서만 해야 하므로
       여기서 하지 않고 분류 단계로 미룬다.
"""
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.signal
from mne_icalabel.iclabel import iclabel_label_components

from load_data import load_subject, to_raw, all_subjects, SFREQ, CH
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인(단일 출처)

# ICLabel 7클래스 순서(문서 명시).
CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')       # 2
BRAIN = CLASSES.index('brain')   # 0
TAU = 0.5                         # argmax & 확률≥TAU 판정

EPOCH_SEC = 6                     # 확정된 에폭 길이

# 안구 파워비율 대역
LOW_BAND = (0.5, 4.0)
FULL_BAND = (0.5, 40.0)

# 뇌파 밴드(δθαβ). 상대파워 분모 = 이 4대역 파워 합(= 0.5–30Hz, 감마 제외와 일관).
BAND_RANGES = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta']
THETA_I, BETA_I = 1, 3            # TBR = theta/beta 인덱스

# 5부위 채널 그룹(데이터셋 channel_labels 순서 기준).
REGIONS = {
    'frontal':   ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],   # 7
    'central':   ['C3', 'C4', 'Cz'],                             # 3
    'parietal':  ['P3', 'P4', 'P7', 'P8', 'Pz'],                 # 5
    'occipital': ['O1', 'O2'],                                   # 2
    'temporal':  ['T7', 'T8'],                                   # 2
}
REGION_IDX = {r: [CH.index(c) for c in chs] for r, chs in REGIONS.items()}

# 뇌파 특징 열 이름(고정 순서): 부위마다 δθαβ 상대파워 4개 + TBR 1개.
NEURAL_COLS = []
for _r in REGIONS:
    NEURAL_COLS += [f'{_r}_{_b}' for _b in BAND_NAMES] + [f'{_r}_tbr']

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / 'exploration' / '_cache'
_RESULTS = _ROOT / 'results'
_OCULAR_CSV = _RESULTS / 'features_ocular.csv'
_NEURAL_CSV = _RESULTS / 'features_neural.csv'

SUBJECT_LIMIT = None              # 정수면 앞 N명만(스모크 테스트). None이면 전원.


# ══ 공용 계산 부품 ═════════════════════════════════════════
def make_epochs(sig, fs=SFREQ, epoch_sec=EPOCH_SEC):
    """1D 신호를 epoch_sec 길이로 겹침 없이 자른다(자투리 버림)."""
    n = int(round(fs * epoch_sec))
    k = len(sig) // n
    return [sig[i * n:(i + 1) * n] for i in range(k)]


def make_epochs_2d(data, fs=SFREQ, epoch_sec=EPOCH_SEC):
    """(채널×시간) 배열을 시간축으로 겹침 없이 자른다. 각 조각 (채널×n)."""
    n = int(round(fs * epoch_sec))
    k = data.shape[1] // n
    return [data[:, i * n:(i + 1) * n] for i in range(k)]


def band_power(freqs, psd, lo, hi):
    """[lo, hi] 대역 파워를 사다리꼴 적분으로 구한다."""
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapz(psd[m], freqs[m])


def psd_fft(epoch, fs=SFREQ):
    """1D 에폭의 PSD를 단일 FFT로 (freqs, psd) 반환. DC 제거."""
    epoch = epoch - epoch.mean()
    return scipy.signal.periodogram(epoch, fs=fs, detrend=False)


# ══ 안구 power류 ═══════════════════════════════════════════
def power_ratio(epoch, fs=SFREQ, low=LOW_BAND, full=FULL_BAND):
    """1D 에폭의 저주파 파워비율(0.5–4 / 0.5–40Hz). low⊂full이라 (0,1]."""
    f, pxx = psd_fft(epoch, fs)
    return band_power(f, pxx, *low) / band_power(f, pxx, *full)


def ocular_source(path):
    """top-1 안구 IC(판정 통과분 중 eye 최대) 시계열. 반환 (sig, eye_prob, eligible)."""
    cleaned = asr(car(bandpass(to_raw(load_subject(path)))))
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)
    eligible_mask = (proba.argmax(axis=1) == EYE) & (proba[:, EYE] >= TAU)
    if not eligible_mask.any():
        return None, float(proba[:, EYE].max()), False
    idx = np.where(eligible_mask)[0]
    top = idx[proba[idx, EYE].argmax()]
    sig = ic.get_sources(cleaned).get_data()[top]
    return sig, float(proba[top, EYE]), True


def ocular_power_features(name, path, label):
    """피험자의 에폭별 파워비율 (n_ep,1)과 메타. 캐시 있으면 로드."""
    cache = _CACHE / f'feat_ocular_{name}.npz'
    if cache.exists():
        d = np.load(cache)
        return {'name': name, 'label': int(d['label']), 'eligible': bool(d['eligible']),
                'eye_prob': float(d['eye_prob']), 'feats': d['feats']}
    sig, eye_prob, eligible = ocular_source(path)
    feats = np.empty((0, 1)) if not eligible else \
        np.array([[power_ratio(e)] for e in make_epochs(sig)])
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, label=np.int64(label),
             eligible=np.bool_(eligible), eye_prob=np.float64(eye_prob))
    return {'name': name, 'label': label, 'eligible': eligible,
            'eye_prob': eye_prob, 'feats': feats}


def variability_from_ratios(ratios):
    """에폭별 파워비율(1D) → 시간 변동성 (cv, succ_diff). 사람당 1세트.

    ADHD의 순간 요동 가설을 겨냥한 event-free 변동성. 사건(깜빡임) 검출 없이
    이미 뽑은 파워비율의 시간적 흩어짐을 잰다(탐지 문제 우회, 전원 값 있음).
      - cv        : std/mean. 전체적 흩어짐(척도 불변).
      - succ_diff : 이웃 에폭 간 |변화|의 평균. 순간순간 급변(moment-to-moment).
    z-정규화는 분류 단계 scaler가 하므로 succ_diff를 mean으로 안 나눠도 됨.
    """
    r = np.asarray(ratios, float).ravel()
    mean = r.mean()
    cv = float(r.std() / mean) if mean > 0 else 0.0
    succ = float(np.mean(np.abs(np.diff(r)))) if len(r) > 1 else 0.0
    return cv, succ


# ══ 뇌파(back-projection → 부위별 밴드파워) ════════════════
def neural_feature_vector(epoch_data, fs=SFREQ):
    """복원된 (19채널 × 시간) 에폭 하나 → 뇌파 특징 25개.

    채널마다 δθαβ 밴드파워를 구하고, 부위별로 채널 평균 낸 뒤 상대화(분모=4대역합).
    부위마다 상대파워 4개 + TBR(θ/β) 1개. 순서는 NEURAL_COLS와 일치.
    """
    # 1) 채널별 밴드파워 (19 × 4)
    chan_bp = np.empty((epoch_data.shape[0], len(BAND_RANGES)))
    for ch in range(epoch_data.shape[0]):
        f, pxx = psd_fft(epoch_data[ch], fs)
        chan_bp[ch] = [band_power(f, pxx, lo, hi) for lo, hi in BAND_RANGES]

    # 2) 부위별: 채널 평균(절대) → 상대화 + TBR
    feats = []
    for idx in REGION_IDX.values():
        region_bp = chan_bp[idx].mean(axis=0)      # (4,) 부위 평균 절대파워
        rel = region_bp / region_bp.sum()          # 4대역합으로 상대화 → 합=1
        tbr = region_bp[THETA_I] / region_bp[BETA_I]
        feats.extend(rel.tolist())
        feats.append(tbr)
    return np.array(feats)                          # (25,)


def neural_source(path):
    """brain IC만 back-projection한 19채널 신경신호. 반환 (data(19×T), n_brain, eligible)."""
    cleaned = asr(car(bandpass(to_raw(load_subject(path)))))
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)
    brain_mask = (proba.argmax(axis=1) == BRAIN) & (proba[:, BRAIN] >= TAU)
    if not brain_mask.any():
        return None, 0, False
    brain_ics = np.where(brain_mask)[0]
    exclude = [i for i in range(proba.shape[0]) if i not in brain_ics]  # brain 외 전부 제거
    recon = ic.apply(cleaned.copy(), exclude=exclude)                    # brain만 남겨 복원
    return recon.get_data(), len(brain_ics), True


def neural_features(name, path, label):
    """피험자의 에폭별 뇌파 특징 (n_ep,25)과 메타. 캐시 있으면 로드."""
    cache = _CACHE / f'feat_neural_{name}.npz'
    if cache.exists():
        d = np.load(cache)
        return {'name': name, 'label': int(d['label']), 'eligible': bool(d['eligible']),
                'n_brain': int(d['n_brain']), 'feats': d['feats']}
    data, n_brain, eligible = neural_source(path)
    feats = np.empty((0, len(NEURAL_COLS))) if not eligible else \
        np.array([neural_feature_vector(e) for e in make_epochs_2d(data)])
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, label=np.int64(label),
             eligible=np.bool_(eligible), n_brain=np.int64(n_brain))
    return {'name': name, 'label': label, 'eligible': eligible,
            'n_brain': n_brain, 'feats': feats}


# ══ 실행: 조건별 표 조립 + 검증 ════════════════════════════
def run_ocular(subjects):
    print('=== 안구 파워비율 특징추출 ===')
    rows, n_ok = [], 0
    for i, (path, label) in enumerate(subjects, 1):
        r = ocular_power_features(path.stem, path, label)
        tag = '' if r['eligible'] else '  ← ocular 불가(제외)'
        print(f'[{i:3d}/{len(subjects)}] {path.stem:<7} eye={r["eye_prob"]:.2f}'
              f'  에폭 {len(r["feats"]):>3}{tag}')
        if not r['eligible']:
            continue
        n_ok += 1
        # 변동성(cv·succ_diff)은 사람당 1세트 → 그 사람 모든 에폭에 복제.
        # subject-wise CV라 복제가 누수를 만들지 않음(같은 사람은 한 fold에만).
        cv, succ = variability_from_ratios(r['feats'][:, 0])
        rows += [(path.stem, int(r['label']), float(e[0]), cv, succ) for e in r['feats']]

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OCULAR_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'label', 'ocular_power_ratio', 'ocular_cv', 'ocular_succ_diff'])
        w.writerows(rows)

    vals = np.array([r[2] for r in rows]) if rows else np.array([])
    cvs = np.array([r[3] for r in rows]) if rows else np.array([])
    succs = np.array([r[4] for r in rows]) if rows else np.array([])
    print(f'\n[안구 검증]  ocular {n_ok}명(84 기대) · 에폭 {len(rows)} · '
          f'NaN {int(np.isnan(vals).sum())}')
    print(f'  파워비율 {vals.min():.3f}~{vals.max():.3f} · '
          f'cv {cvs.min():.3f}~{cvs.max():.3f} · succ_diff {succs.min():.3f}~{succs.max():.3f}')
    print(f'  저장: {_OCULAR_CSV}\n')


def run_neural(subjects):
    print('=== 뇌파 부위별 밴드파워 특징추출 ===')
    rows, n_ok, n_brains = [], 0, []
    for i, (path, label) in enumerate(subjects, 1):
        r = neural_features(path.stem, path, label)
        tag = '' if r['eligible'] else '  ← brain IC 없음(제외)'
        print(f'[{i:3d}/{len(subjects)}] {path.stem:<7} brain IC={r["n_brain"]:>2}'
              f'  에폭 {len(r["feats"]):>3}{tag}')
        if not r['eligible']:
            continue
        n_ok += 1
        n_brains.append(r['n_brain'])
        rows += [(path.stem, int(r['label']), *map(float, e)) for e in r['feats']]

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_NEURAL_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'label', *NEURAL_COLS])
        w.writerows(rows)

    arr = np.array([r[2:] for r in rows]) if rows else np.empty((0, len(NEURAL_COLS)))
    print(f'\n[뇌파 검증]  brain 보유 {n_ok}명(121 기대) · 에폭 {len(rows)} · '
          f'NaN {int(np.isnan(arr).sum())}')
    if len(arr):
        for ri, rname in enumerate(REGIONS):
            s = arr[:, ri * 5:ri * 5 + 4].sum(axis=1)   # 그 부위 상대파워 4개 합
            print(f'  {rname:<9} 상대합 {s.min():.3f}~{s.max():.3f}(1 기대) · '
                  f'TBR {arr[:, ri*5+4].min():.2f}~{arr[:, ri*5+4].max():.2f}')
        print(f'  brain IC 개수: 최소 {min(n_brains)} · 최대 {max(n_brains)} · '
              f'평균 {np.mean(n_brains):.1f}')
    print(f'  저장: {_NEURAL_CSV}')


if __name__ == '__main__':
    subjects = all_subjects()
    if SUBJECT_LIMIT is not None:
        subjects = subjects[:SUBJECT_LIMIT]
    run_ocular(subjects)
    run_neural(subjects)