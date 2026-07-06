"""에폭 길이 × PSD 추정기 안정성 스윕(진단 전용, 일회성).

목적 : 안구 주력 특징인 저주파 파워비율(0.5–4Hz / 0.5–40Hz)을 에폭 단위로 잴 때,
       어떤 (에폭 길이 × 추정기) 조합이 가장 안정적으로 재는지 경험적으로 찾는다.
       파워비율만이 유일한 에폭 단위 특징이라(blink rate는 피험자 단위) 에폭 길이는
       이 특징 하나로 정한다.

무엇을 재나(안정성) :
  - split-half 재현성(주) : 홀수 에폭 묶음 vs 짝수 에폭 묶음의 파워비율 평균 차이.
                            |홀평균 − 짝평균| / 전체평균. 작을수록 재현성 높음.
                            추정 잡음을 겨냥(진짜 순간 변동은 양 묶음에 고루 들어가 상쇄).
  - 변동계수 CV(보조)     : 에폭 간 파워비율의 std/mean. 작을수록 안정.
  두 지표 모두 피험자 안에서 계산 후 6명 평균(작은 표본에서 상관 대신 피험자 내 지표).

축 :
  - 에폭 길이 : 4 / 6 / 8초 (겹침 없음, 자투리 버림)
  - 추정기    : 단일 FFT(주기도표) / multitaper / Welch(대조군, nperseg=에폭 절반)
  - 대표 6명  : 확률·길이 양극단을 덮게 84명에서 선정
                (뚜렷×긴: v20p, v300 / 뚜렷×짧은: v254, v46p / 경계선: v1p, v149)

집계 정합 : 파워비율은 집계 방식 B(top-1 대표 안구 IC)로 뽑기로 확정됨. 그래서 이
       진단도 각 피험자의 top-1 안구 IC(eye argmax) 하나에서 파워비율을 잰다 —
       실제 특징추출을 그대로 미러링.

누수 경계 : 그룹 라벨·분류 정확도를 보지 않는다. 신호 수준 안정성만 본다.

전처리 재사용 : bandpass → CAR → ASR(100) → ICA를 preprocess.py에서 그대로 import.
       캐시엔 IC 시계열이 없어(확률만) 6명은 파이프라인을 다시 돌려 IC 신호를 얻는다
       (random_state 고정이라 캐시와 동일 분해 → top-1 안구 IC 일치).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import scipy.signal
from mne.time_frequency import psd_array_multitaper
from mne_icalabel.iclabel import iclabel_label_components

from load_data import load_subject, to_raw, all_subjects, SFREQ
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인(단일 출처)

# ICLabel 7클래스 순서(문서 명시). eye = 인덱스 2.
CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')

# 파워비율 대역
LOW_BAND = (0.5, 4.0)     # 분자: 저주파(안구가 몰리는 대역)
FULL_BAND = (0.5, 40.0)   # 분모: 분석 전대역

# 스윕 축
SUBJECTS = ['v20p', 'v254', 'v1p', 'v300', 'v46p', 'v149']
EPOCH_SECS = [4, 6, 8]
ESTIMATORS = ['fft', 'mt', 'welch']


# ── 파워 적분: 한 대역의 파워를 사다리꼴로 적분 ──────────────
def band_power(freqs, psd, lo, hi):
    """[lo, hi] 대역의 파워를 사다리꼴 적분으로 구한다.
    세 추정기 모두 같은 방식으로 적분해 비교 가능성을 유지한다.
    """
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapz(psd[m], freqs[m])


# ── 파워비율: 추정기별로 PSD를 구해 저주파/전대역 비율 ───────
def power_ratio(epoch, fs, estimator):
    """한 에폭(1D)의 저주파 파워비율(0.5–4 / 0.5–40Hz)을 추정기별로 계산.

    - fft   : 단일 주기도표(창 전체 한 번). 해상도 최고, 분산 큼.
    - mt    : multitaper. 창은 그대로 두고 taper 여러 개로 평균 → 해상도 유지+분산↓.
    - welch : 에폭을 절반 길이 서브윈도우로 쪼개 평균. 분산↓지만 해상도 나빠짐
              → 0.5Hz 하한에 불리(대조군: 왜 저주파에 지는지 수치로 확인).
    DC는 미리 제거해 세 추정기에 동일 입력을 준다(0.5Hz 하한 밖이라 정보 손실 없음).
    """
    epoch = epoch - epoch.mean()                  # DC 제거(동일 입력 보장)
    if estimator == 'fft':
        f, pxx = scipy.signal.periodogram(epoch, fs=fs, detrend=False)
    elif estimator == 'welch':
        nperseg = len(epoch) // 2                  # 에폭 절반
        f, pxx = scipy.signal.welch(epoch, fs=fs, nperseg=nperseg, detrend=False)
    elif estimator == 'mt':
        pxx, f = psd_array_multitaper(epoch, sfreq=fs, fmin=0, fmax=fs / 2,
                                      verbose=False)
    else:
        raise ValueError(f'알 수 없는 추정기: {estimator}')
    low = band_power(f, pxx, *LOW_BAND)
    full = band_power(f, pxx, *FULL_BAND)
    return low / full


# ── 에폭 분할: 겹침 없이 자르고 자투리는 버림 ────────────────
def make_epochs(sig, fs, epoch_sec):
    """신호(1D)를 epoch_sec 길이로 겹침 없이 자른다. 마지막 자투리는 버린다."""
    n = int(round(fs * epoch_sec))
    k = len(sig) // n
    return [sig[i * n:(i + 1) * n] for i in range(k)]


# ── 피험자 내 안정성 지표: split-half + CV ───────────────────
def stability(ratios):
    """에폭별 파워비율 배열 → (split-half, CV). 둘 다 작을수록 안정.

    split-half : 홀짝(interleaved) 두 묶음의 평균 차이를 전체평균으로 정규화.
                 시간축에 고루 퍼진 두 묶음이라 앞뒤 상태변화 편향이 줄고,
                 진짜 순간 변동은 양쪽에 고루 들어가 상쇄 → 추정 잡음이 드러남.
    """
    ratios = np.asarray(ratios)
    mean = ratios.mean()
    cv = ratios.std() / mean
    odd = ratios[0::2].mean()      # 1,3,5...번째(0-based 0,2,4)
    even = ratios[1::2].mean()     # 2,4,6...번째(0-based 1,3,5)
    split_half = abs(odd - even) / mean
    return split_half, cv


# ── top-1 안구 IC 시계열 추출: 전처리~ICA~ICLabel 후 대표 IC ──
def ocular_source(path):
    """한 피험자의 top-1 안구 IC(eye argmax) 시계열(1D)과 그 eye 확률을 반환.
    파이프라인은 preprocess.py 잠근 함수를 그대로 재사용(단일 출처).
    """
    cleaned = asr(car(bandpass(to_raw(load_subject(path)))))
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)     # (n_comp, 7)
    top = int(proba[:, EYE].argmax())                 # 가장 안구다운 IC
    sig = ic.get_sources(cleaned).get_data()[top]     # 그 IC 시계열(1D)
    return sig, float(proba[top, EYE])


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    fs = SFREQ
    path_of = {p.stem: p for p, _ in all_subjects()}

    # 조합별 (피험자별 지표) 누적: results[(sec, est)] = [(sh, cv), ...]
    results = {(sec, est): [] for sec in EPOCH_SECS for est in ESTIMATORS}

    print('=== 에폭 길이 × 추정기 안정성 스윕 ===')
    print(f'대표 {len(SUBJECTS)}명: {", ".join(SUBJECTS)}\n')
    print('[피험자별 top-1 안구 IC + 길이별 에폭 수]')
    print(f'{"이름":<7}{"eye확률":>7}{"길이(s)":>9}'
          f'{"에폭@4":>7}{"에폭@6":>7}{"에폭@8":>7}')
    print('-' * 44)

    for name in SUBJECTS:
        sig, eye_p = ocular_source(path_of[name])
        dur = len(sig) / fs
        n_ep = {sec: len(make_epochs(sig, fs, sec)) for sec in EPOCH_SECS}
        print(f'{name:<7}{eye_p:>7.2f}{dur:>9.1f}'
              f'{n_ep[4]:>7}{n_ep[6]:>7}{n_ep[8]:>7}')

        for sec in EPOCH_SECS:
            epochs = make_epochs(sig, fs, sec)
            if len(epochs) < 2:              # split-half 불가(자투리만)
                continue
            for est in ESTIMATORS:
                ratios = [power_ratio(e, fs, est) for e in epochs]
                results[(sec, est)].append(stability(ratios))

    # ── 조합별 6명 평균 표 ──
    print('\n[조합별 안정성 — 6명 평균]  (split-half·CV 둘 다 낮을수록 안정)')
    print(f'{"길이":<6}{"추정기":<10}{"split-half":>12}{"CV":>10}{"n명":>5}')
    print('-' * 43)
    for sec in EPOCH_SECS:
        for est in ESTIMATORS:
            vals = np.array(results[(sec, est)])   # (n_subj, 2)
            if len(vals) == 0:
                continue
            sh_m = vals[:, 0].mean()
            cv_m = vals[:, 1].mean()
            print(f'{str(sec) + "s":<6}{est:<10}{sh_m:>12.4f}{cv_m:>10.4f}'
                  f'{len(vals):>5}')

    print('\n해석 : split-half·CV가 함께 낮은 조합이 파워비율을 가장 안정적으로 잰다.')
    print('       짧은 녹화(v46p·v254·v1p)에서 8초가 조각 부족으로 흔들리는지 특히 확인.')
    print('       Welch가 저주파 해상도 손실로 뒤처지는지도 대조.')