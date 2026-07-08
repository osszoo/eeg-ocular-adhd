"""특징추출 파이프라인(정식 출처).

전처리(preprocess.py)까지 끝난 신호에서 분류용 특징을 뽑아 (에폭 × 특징) 표로
만든다. 진단 스크립트(exploration/)가 검증한 로직을 여기서 정식 함수로 승격한다 —
앞으로 특징 계산의 single source of truth는 이 파일이다.

설계(확정):
  - 에폭 : 6초, 겹침 없음 (epoch_estimator_sweep로 확정)
  - 추정기 : 단일 FFT(주기도표) (동 진단으로 확정)
  - 표 단위 : 한 행 = 6초 에폭 하나. 한 피험자는 여러 행.
              CV는 subject_id로 묶어 분할(에폭이 train/test에 안 섞이게).

이번 파일이 담는 것(D 첫 조각):
  - 안구 power류 = 저주파 파워비율(0.5–4 / 0.5–40Hz), top-1 안구 IC(집계 B), 84명 전원.
  - 뇌파(부위별 밴드파워)·안구 rate류는 이 뼈대 위에 이후 얹는다.

캐시·산출물:
  - 피험자별 캐시(기계장치) : exploration/_cache/feat_ocular_{name}.npz
        무거운 전처리~ICA~특징 계산을 피험자당 한 번만. 재실행 시 건너뜀.
  - 최종 표(산출물)          : results/features_ocular.csv
        subject_id·label·특징 열. 사람이 열어 검증하고 sklearn에 먹인다.

누수 경계 : 특징 계산은 그룹 라벨을 보지 않는다. 84명 판정도 라벨이 아니라
       안구 IC 기준(eye argmax & ≥0.5)으로만 한다. 표에 label 열을 붙이는 건
       이후 지도학습·CV용이며 특징 계산에 관여하지 않는다.
       z-정규화는 여기서 하지 않는다 — train fold에서만 계산해야 하므로 분류 단계로 미룸.
"""
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.signal
from mne_icalabel.iclabel import iclabel_label_components

from load_data import load_subject, to_raw, all_subjects, SFREQ
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인(단일 출처)

# ICLabel 7클래스 순서(문서 명시). eye = 인덱스 2.
CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')
TAU = 0.5                      # ocular 판정 임계(argmax & eye≥TAU)

EPOCH_SEC = 6                  # 확정된 에폭 길이
LOW_BAND = (0.5, 4.0)         # 파워비율 분자(안구 저주파)
FULL_BAND = (0.5, 40.0)      # 파워비율 분모(분석 전대역)

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / 'exploration' / '_cache'          # 기존 gitignore된 캐시 위치 재사용
_RESULTS = _ROOT / 'results'
_OCULAR_CSV = _RESULTS / 'features_ocular.csv'

# 스모크 테스트용: 정수로 두면 앞 N명만 처리(뼈대 점검). None이면 전원.
SUBJECT_LIMIT = None


# ── 에폭 분할: 겹침 없이 자르고 자투리는 버림 ────────────────
def make_epochs(sig, fs=SFREQ, epoch_sec=EPOCH_SEC):
    """신호(1D)를 epoch_sec 길이로 겹침 없이 자른다. 마지막 자투리는 버린다."""
    n = int(round(fs * epoch_sec))
    k = len(sig) // n
    return [sig[i * n:(i + 1) * n] for i in range(k)]


# ── 파워 적분: 한 대역의 파워를 사다리꼴로 적분 ──────────────
def band_power(freqs, psd, lo, hi):
    """[lo, hi] 대역의 파워를 사다리꼴 적분으로 구한다."""
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapz(psd[m], freqs[m])


# ── PSD: 단일 FFT 주기도표(확정 추정기) ──────────────────────
def psd_fft(epoch, fs=SFREQ):
    """한 에폭(1D)의 PSD를 단일 FFT로 구해 (freqs, psd) 반환. DC 제거."""
    epoch = epoch - epoch.mean()
    return scipy.signal.periodogram(epoch, fs=fs, detrend=False)


# ── 안구 power류: 저주파 파워비율 ────────────────────────────
def power_ratio(epoch, fs=SFREQ, low=LOW_BAND, full=FULL_BAND):
    """한 에폭의 저주파 파워비율(0.5–4 / 0.5–40Hz). low⊂full이라 값은 (0,1]."""
    f, pxx = psd_fft(epoch, fs)
    return band_power(f, pxx, *low) / band_power(f, pxx, *full)


# ── top-1 안구 IC 시계열: 전처리~ICA~ICLabel 후 대표 안구 IC ──
def ocular_source(path):
    """한 피험자의 top-1 안구 IC(ocular 판정 통과분 중 eye 확률 최대) 시계열을 반환.

    반환: (sig, eye_prob, eligible)
      - eligible=False : eye argmax & ≥TAU IC가 없음(ocular 조건 불가). sig=None.
    파이프라인은 preprocess.py 잠근 함수를 그대로 재사용(단일 출처).
    random_state 고정이라 iclabel 캐시(ocular_survey)와 동일 분해.
    """
    cleaned = asr(car(bandpass(to_raw(load_subject(path)))))
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)          # (n_comp, 7)

    is_eye_argmax = proba.argmax(axis=1) == EYE
    eligible_mask = is_eye_argmax & (proba[:, EYE] >= TAU)  # ocular_survey와 동일 기준
    if not eligible_mask.any():
        return None, float(proba[:, EYE].max()), False

    eligible_idx = np.where(eligible_mask)[0]
    top = eligible_idx[proba[eligible_idx, EYE].argmax()]  # 판정 통과분 중 eye 최대
    sig = ic.get_sources(cleaned).get_data()[top]          # 그 IC 시계열(1D)
    return sig, float(proba[top, EYE]), True


# ── 피험자 1명 안구 power 특징(에폭 × 1). 캐시 재사용 ────────
def ocular_power_features(name, path, label):
    """피험자의 에폭별 파워비율 (n_ep, 1)과 메타를 반환. 캐시 있으면 로드.

    반환 dict: {name, label, eligible, eye_prob, feats(n_ep,1)}
    """
    cache = _CACHE / f'feat_ocular_{name}.npz'
    if cache.exists():
        d = np.load(cache)
        return {'name': name, 'label': int(d['label']),
                'eligible': bool(d['eligible']), 'eye_prob': float(d['eye_prob']),
                'feats': d['feats']}

    sig, eye_prob, eligible = ocular_source(path)
    if not eligible:
        feats = np.empty((0, 1))
    else:
        epochs = make_epochs(sig)
        feats = np.array([[power_ratio(e)] for e in epochs])   # (n_ep, 1)

    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, label=np.int64(label),
             eligible=np.bool_(eligible), eye_prob=np.float64(eye_prob))
    return {'name': name, 'label': label, 'eligible': eligible,
            'eye_prob': eye_prob, 'feats': feats}


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    subjects = all_subjects()
    if SUBJECT_LIMIT is not None:
        subjects = subjects[:SUBJECT_LIMIT]

    print('=== 안구 파워비율 특징추출 (D 첫 조각) ===')
    print(f'대상 후보: {len(subjects)}명  (에폭 {EPOCH_SEC}s, 단일 FFT, '
          f'비율 {LOW_BAND[0]}–{LOW_BAND[1]} / {FULL_BAND[0]}–{FULL_BAND[1]}Hz)\n')

    rows = []                 # (subject_id, label, ocular_power_ratio) 에폭 단위
    n_eligible = 0
    for i, (path, label) in enumerate(subjects, 1):
        name = path.stem
        r = ocular_power_features(name, path, label)
        tag = '' if r['eligible'] else '  ← ocular 불가(제외)'
        n_ep = len(r['feats'])
        print(f'[{i:3d}/{len(subjects)}] {name:<7} '
              f'eye={r["eye_prob"]:.2f}  에폭 {n_ep:>3}{tag}')
        if not r['eligible']:
            continue
        n_eligible += 1
        for e in r['feats']:
            rows.append((name, int(r['label']), float(e[0])))

    # ── 최종 표 조립 → csv ──
    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OCULAR_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'label', 'ocular_power_ratio'])
        w.writerows(rows)

    # ── 검증 ──
    vals = np.array([r[2] for r in rows]) if rows else np.array([])
    print('\n[검증]')
    print(f'  ocular 보유(84 기대): {n_eligible}명')
    print(f'  총 에폭(행) 수      : {len(rows)}')
    print(f'  NaN 개수(0 기대)    : {int(np.isnan(vals).sum())}')
    if len(vals):
        print(f'  파워비율 범위       : {vals.min():.3f} ~ {vals.max():.3f}  '
              f'(0<r≤1 이어야 정상)')
        if vals.max() > 1.0:
            print('  ⚠ 1을 넘는 값 존재 — 대역 적분 버그 의심')
    print(f'  저장: {_OCULAR_CSV}')