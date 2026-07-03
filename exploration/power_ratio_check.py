"""저주파 파워비율 검증(진단 전용, 일회성): 이 특징이 안구 IC를 구별하나.

주력 안구 특징 후보 = 저주파 파워비율 = (0.5~4Hz 파워) / (0.5~40Hz 전체 파워).
안구 활동(깜빡임·안구운동)은 저주파에 몰리므로, 진짜 안구 IC일수록 이 비율이
높아야 한다. 비율이라 신호 스케일에 불변(단위 불확정 데이터에 안전).

검증 질문 : 이 비율이 안구 IC(eye 확률 높음)에서 높고, 비안구 IC에서 낮은가?
   그렇다면 특징이 안구성을 실제로 포착 → 주력 특징으로 채택.
   안구/비안구가 비슷하면 판별력 없음 → 재설계.

방법 : 6명(pick_subjects)의 모든 IC에 대해 (eye 확률, 저주파 파워비율)을 구해
   산점도로 본다. x=eye 확률, y=파워비율. 오른쪽 위로 쏠리면(eye 높을수록 비율
   높음) 특징 유효. eye≥0.5인 IC(우리 ocular 기준)를 색으로 강조.

   rate가 아니라 PSD 적분 비율이므로 깜빡임 형태(뚜렷/연속)와 무관하게 계산됨 —
   앞서 blink_survey에서 본 '연속 진동' 피험자에서도 값이 나온다.

파이프라인은 잠근 최종본. 라벨(ADHD/Control) 미사용 → 누수 아님.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_icalabel.iclabel import iclabel_label_components
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr, ica, ASR_CUTOFF
from asr_sweep import pick_subjects

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
mne.set_log_level('WARNING')

CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')
LOW = (0.5, 4.0)      # 저주파(안구) 대역 = 분자
FULL = (0.5, 40.0)    # 전체 대역 = 분모 (파이프라인 대역)


def low_freq_ratio(sources):
    """각 IC의 저주파 파워비율 배열 (n_comp,). PSD 적분 비율."""
    # IC 소스는 채널 타입이 'misc'라 기본 picks가 못 잡음 → 전체 채널 명시
    psd = sources.compute_psd(fmin=FULL[0], fmax=FULL[1], picks='all')
    power = psd.get_data(picks='all')   # (n_comp, n_freqs) — 파워
    freqs = psd.freqs
    low = (freqs >= LOW[0]) & (freqs <= LOW[1])
    full = (freqs >= FULL[0]) & (freqs <= FULL[1])
    return power[:, low].sum(axis=1) / power[:, full].sum(axis=1)


# 6명: ADHD·Control·짧은 녹화 포함
sample = []
sample += [(f, 'ADHD')    for f, d in pick_subjects(['ADHD_part1', 'ADHD_part2'])]
sample += [(f, 'Control') for f, d in pick_subjects(['Control_part1', 'Control_part2'])]

all_eye, all_ratio = [], []      # 전체 IC 모음(산점도용)
print(f'저주파 파워비율 = {LOW[0]}~{LOW[1]}Hz / {FULL[0]}~{FULL[1]}Hz\n')
for f, grp in sample:
    cleaned = asr(car(bandpass(to_raw(load_subject(f)))), cutoff=ASR_CUTOFF)
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)
    eye = proba[:, EYE]
    sources = ic.get_sources(cleaned)
    ratio = low_freq_ratio(sources)
    all_eye.extend(eye.tolist())
    all_ratio.extend(ratio.tolist())
    # 대표 안구 IC(eye 최고)의 비율 vs 그 피험자 비안구 IC 비율 중앙값
    top = int(eye.argmax())
    nonocular = ratio[np.arange(len(eye)) != top]
    print(f'{f.stem:6s}({grp:7s}) 안구IC=IC{top:02d} eye={eye[top]:.2f} '
          f'비율={ratio[top]:.3f}  |  비안구 비율 중앙값={np.median(nonocular):.3f}')

all_eye = np.array(all_eye)
all_ratio = np.array(all_ratio)

# --- 산점도: x=eye 확률, y=저주파 파워비율 ---
fig, ax = plt.subplots(figsize=(9, 6))
lo = all_eye < 0.5
hi = all_eye >= 0.5
ax.scatter(all_eye[lo], all_ratio[lo], s=25, c='tab:gray', alpha=0.6, label='eye < 0.5 (비안구/애매)')
ax.scatter(all_eye[hi], all_ratio[hi], s=40, c='tab:red', alpha=0.8, label='eye >= 0.5 (ocular 기준)')
ax.axvline(0.5, color='tab:red', ls='--', lw=0.8)
ax.set(xlabel='ICLabel eye 확률', ylabel='저주파 파워비율 (0.5~4 / 0.5~40Hz)',
       title='저주파 파워비율 vs eye 확률 (6명 전체 IC)\n오른쪽 위로 쏠리면 특징이 안구성을 포착')
ax.legend()
plt.tight_layout()
plt.show()