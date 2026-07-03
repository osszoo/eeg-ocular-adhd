"""깜빡임 검출 시험 B안+저주파 필터(진단 전용, 일회성).

경과 :
  A안(MNE find_eog_events) → 271개, rate 2.4/s. EOG용이라 IC 스케일에 안 맞아 과검출. 폐기.
  B안(SD 배수 피크, 넓은 대역) → N=4.0에서도 rate 0.78/s로 과검출.
    원인: IC02가 0.5~40Hz 넓은 대역이라 느린 깜빡임 언덕 위에 빠른 성분이 얹혀
    신호가 내내 뾰족하게 진동 → SD로 잘라도 큰 봉우리가 너무 많음.

이번 : 깜빡임은 저주파(완만한 언덕)이므로, "검출 전에만" IC02를 저주파로 거른 뒤
       SD 배수 피크를 찾는다. A안 find_eog_events도 내부에서 1~10Hz로 걸러 검출했던
       것과 같은 원리. 이 저주파 필터는 '검출용 돋보기'일 뿐 — 전처리 신호나 특징
       계산에 쓰는 신호를 바꾸지 않는다(찾은 '시점'만 사용).

       여러 대역을 비교: 어디서 깜빡임이 드문드문한 언덕으로 깨끗이 드러나는지.
       판정 기준은 분류 정확도가 아니라 생리적 타당성(rate 0.2~0.4/s) → 누수 아님.

변수 고정 : v10p 깜빡임 IC = IC02(확정). 부호는 ICA 임의 → 절댓값으로 크기만 본다.
v10p 한 명이라 빠르다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr, ica, ASR_CUTOFF

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
mne.set_log_level('WARNING')

_ROOT = Path(__file__).resolve().parent.parent
SUBJ = _ROOT / 'ADHD_part1' / 'v10p.mat'
BLINK_IC = 2
MIN_SEP_S = 0.3          # 최소 간격(초): 깜빡임 최소 간격. 넓은 대역보다 약간 크게.
N_SD = 3.0               # SD 배수(고정) — 대역 효과를 보려고 N은 3.0으로 고정
BANDS = [None, (1, 10), (1, 5), (0.5, 4), (1, 4)]   # None = 필터 안 함(넓은 대역, 비교 기준)

# --- 잠근 파이프라인: bandpass → CAR → ASR(100) → ICA ---
cleaned = asr(car(bandpass(to_raw(load_subject(SUBJ)))), cutoff=ASR_CUTOFF)
ic = ica(cleaned)
sources = ic.get_sources(cleaned)
sfreq = sources.info['sfreq']
dur = sources.n_times / sfreq
raw_sig = sources.get_data(picks=[sources.ch_names[BLINK_IC]])[0]   # IC02 원본(넓은 대역)
min_sep = int(MIN_SEP_S * sfreq)


def lowband(sig, band):
    """검출용 저주파 필터. band=None이면 원본 그대로."""
    if band is None:
        return sig.copy()
    lo, hi = band
    return mne.filter.filter_data(sig.astype(float), sfreq, lo, hi, verbose=False)


def detect(sig):
    """평균 제거·절댓값 후 N_SD×SD 넘는 피크."""
    a = np.abs(sig - sig.mean())
    peaks, _ = find_peaks(a, height=N_SD * a.std(), distance=min_sep)
    return peaks


# --- 대역별 비교 (N_SD 고정) ---
print(f'=== v10p IC{BLINK_IC:02d} 깜빡임 검출: 검출용 저주파 대역 비교 (N={N_SD}) ===')
print(f'녹화 {dur:.1f}s,  최소간격 {MIN_SEP_S}s')
print(f'{"검출 대역":>12}{"검출수":>8}{"rate(/s)":>10}{"간격CV":>8}   생리적 0.2~0.4/s')
print('-' * 54)
results = {}
for band in BANDS:
    sig = lowband(raw_sig, band)
    pk = detect(sig)
    results[band] = (sig, pk)
    rate = len(pk) / dur
    if len(pk) >= 3:
        iv = np.diff(pk / sfreq)
        cv_s = f'{iv.std()/iv.mean():.3f}'
    else:
        cv_s = '  -'
    name = '필터안함' if band is None else f'{band[0]}~{band[1]}Hz'
    flag = '  <- 그럴듯' if 0.2 <= rate <= 0.4 else ''
    print(f'{name:>12}{len(pk):>8}{rate:>10.3f}{cv_s:>8}{flag}')

# --- 대표 대역으로 시각화 (범위에 든 첫 대역, 없으면 1~5Hz) ---
rep = next((b for b in BANDS if b is not None
            and 0.2 <= len(results[b][1]) / dur <= 0.4), (1, 5))
sig, peaks = results[rep] if rep in results else (lowband(raw_sig, rep), detect(lowband(raw_sig, rep)))
peak_times = peaks / sfreq
t = np.arange(len(raw_sig)) / sfreq
rep_name = f'{rep[0]}~{rep[1]}Hz'

print(f'\n대표 대역 = {rep_name} ({len(peaks)}개, rate {len(peaks)/dur:.3f}/s)')
if len(peaks) >= 3:
    iv = np.diff(peak_times)
    print(f'  간격 CV {iv.std()/iv.mean():.3f}  (평균 {iv.mean():.2f}s, 표준편차 {iv.std():.2f}s)')

# 위: 원본(넓은 대역) + 검출점 / 아래: 저주파 필터된 신호 + 검출점
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax1.plot(t, raw_sig, lw=0.4, color='tab:blue')
ax1.plot(peak_times, raw_sig[peaks], 'rx', ms=7)
ax1.set(ylabel='IC 진폭 (a.u.)', title=f'v10p IC{BLINK_IC:02d} 원본(넓은 대역) — 검출점 표시 ({len(peaks)}개)')

sd = np.abs(sig - sig.mean()).std()
ax2.plot(t, sig, lw=0.6, color='tab:green')
ax2.plot(peak_times, sig[peaks], 'rx', ms=7, label=f'검출 ({rep_name}, N={N_SD})')
ax2.axhline(sig.mean() + N_SD * sd, color='gray', ls='--', lw=0.7)
ax2.axhline(sig.mean() - N_SD * sd, color='gray', ls='--', lw=0.7)
ax2.set(xlabel='시간 (s)', ylabel='IC 진폭 (a.u.)',
        title=f'검출용 저주파 필터({rep_name}) 적용 — 깜빡임 언덕이 드러나나')
ax2.legend()
plt.tight_layout()
plt.show()