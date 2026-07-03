"""깜빡임 형태 조사(진단 전용, 일회성): 여러 피험자의 안구 IC가 어떻게 생겼나.

배경 : v10p IC02는 저주파 필터 후에도 내내 진동 — "잠잠 + 드문 깜빡임"이 아니라
       지속적 안구활동으로 보였다. 그래서 '깜빡임 rate'가 이 신호에 안 맞았다.
       하지만 v10p 하나로 rate를 폐기하긴 이르다. v10p가 예외인가 전형인가?

목적 : pick_subjects 6명(ADHD·Control·짧은 녹화 포함)의 대표 안구 IC 시계열을
       눈으로 비교한다. 대부분이 v10p처럼 연속 진동이면 rate를 접고 power류로,
       상당수가 드문 깜빡임 언덕이면 rate를 살리되 검출을 다듬는다.

대표 안구 IC 선택 : 피험자마다 IC 순서가 다르므로 IC02로 못 박지 못한다.
       ICLabel eye 확률이 최고인 IC를 그 사람의 대표 안구 IC로 잡는다(곧 쓸
       분류 기준 eye argmax와 일치). eye 확률도 함께 표기 — 낮으면 애매한 것.

표시 : 피험자별 2단(원본 넓은 대역 / 검출용 1~5Hz 필터)을 세로로 늘어놓아
       "필터가 깜빡임 언덕을 드러내나"까지 한눈에. print 없이 그림 위주.
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
from asr_sweep import pick_subjects   # 6명 선정 재사용(단일 출처, __main__ 가드 있음)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
mne.set_log_level('WARNING')

CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')
DETECT_BAND = (1, 5)   # 검출용 저주파 대역(앞 시험에서 그나마 나았던 값)

# 6명: asr_sweep과 동일 (ADHD·Control·짧은 녹화 + v265 이상치)
sample = []
sample += [(f, 'ADHD')    for f, d in pick_subjects(['ADHD_part1', 'ADHD_part2'])]
sample += [(f, 'Control') for f, d in pick_subjects(['Control_part1', 'Control_part2'])]

# 각 피험자: 대표 안구 IC 시계열 + eye 확률 확보
rows = []
for f, grp in sample:
    cleaned = asr(car(bandpass(to_raw(load_subject(f)))), cutoff=ASR_CUTOFF)
    ic = ica(cleaned)
    proba = iclabel_label_components(cleaned, ic)   # (n_comp, 7)
    eye = proba[:, EYE]
    idx = int(eye.argmax())                         # eye 확률 최고 IC
    sources = ic.get_sources(cleaned)
    sig = sources.get_data(picks=[sources.ch_names[idx]])[0]
    sfreq = sources.info['sfreq']
    filt = mne.filter.filter_data(sig.astype(float), sfreq, DETECT_BAND[0], DETECT_BAND[1],
                                  verbose=False)
    rows.append({'name': f.stem, 'grp': grp, 'ic': idx, 'eye': eye[idx],
                 'sig': sig, 'filt': filt, 'sfreq': sfreq})
    print(f'{f.stem:6s} ({grp:7s}) 대표 안구 IC = IC{idx:02d}, eye={eye[idx]:.2f}')

# --- 시각화: 피험자당 2행(원본 / 필터), 6명 = 12행 ---
n = len(rows)
fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n))
for i, r in enumerate(rows):
    t = np.arange(len(r['sig'])) / r['sfreq']
    tag = f"{r['name']} ({r['grp']}) IC{r['ic']:02d} eye={r['eye']:.2f}"
    axes[i][0].plot(t, r['sig'], lw=0.4, color='tab:blue')
    axes[i][0].set_ylabel(tag, fontsize=8)
    if i == 0:
        axes[i][0].set_title('원본 (넓은 대역 0.5~40Hz)')
    axes[i][1].plot(t, r['filt'], lw=0.5, color='tab:green')
    if i == 0:
        axes[i][1].set_title(f'검출용 저주파 {DETECT_BAND[0]}~{DETECT_BAND[1]}Hz')
axes[-1][0].set_xlabel('시간 (s)')
axes[-1][1].set_xlabel('시간 (s)')
plt.tight_layout()
plt.show()