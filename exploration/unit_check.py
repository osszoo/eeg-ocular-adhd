"""단위(스케일) 진단: .mat 신호가 µV인지 확인하는 일회성 검증.
MNE의 V↔µV 자동 변환을 피하려 scipy로 native 단위에서 직접 측정한다.
채널 순서 교정 이후 재검증 버전 — CH를 load_data에서 import해 순서 불일치를 원천 차단.
세 증거 축을 한 번에 본다:
  ① 전두 깜빡임 피크(Fp1·Fp2): 절대 진폭이 알려진 생리신호로 단위 자릿수 역산
  ② 후두/두정 배경(O1·O2·Pz·P3·P4): 안구 오염 적은 순수 EEG 배경으로 교차검증
  ③ CAR 전후 배경: 귓불(A1·A2) 기준 공통성분이 배경을 부풀렸는지 확인"""
import sys
from pathlib import Path

# src/ 를 경로에 추가 (이 파일은 exploration/ 안에 있고 레포 루트는 그 부모)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from load_data import load_subject, CH   # 로딩·채널순서 단일 출처 (CH 드리프트 차단)

SFREQ = 128

arr = load_subject(sorted(Path('ADHD_part1').glob('*.mat'))[0])   # 한 명 기준

# 1–40Hz 밴드패스: 표류 + 전원잡음 제거, native 단위 유지
b, a = butter(4, [1/(SFREQ/2), 40/(SFREQ/2)], btype='band')
filt = filtfilt(b, a, arr, axis=0)

mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))   # 잡음에 강한 배경 추정

# ① 전두 깜빡임 — 절대 진폭이 알려진 생리신호로 단위 자릿수 역산
print('=== ① 전두 깜빡임 (Fp1·Fp2) ===')
for name in ['Fp1', 'Fp2']:
    x = filt[:, CH.index(name)]
    bg = mad(x)
    peaks, props = find_peaks(x, height=4*bg, distance=int(0.2*SFREQ))
    peak_med = np.median(props['peak_heights']) if len(peaks) else float('nan')
    rate = len(peaks) / (len(x)/SFREQ)
    print(f'  {name}: 배경 {bg:.1f} | 깜빡임 피크 중앙높이 {peak_med:.1f} | 검출율 {rate:.2f}/s')
print('  → 배경 10~30·피크 100~400이면 µV. 배경 100~300·피크 1000~2000이면 약 10배 스케일.\n')

# ② 후두/두정 배경 — 안구 오염 적은 순수 EEG 배경으로 교차검증
print('=== ② 후두/두정 배경 (안구 오염 적음) ===')
for name in ['O1', 'O2', 'Pz', 'P3', 'P4']:
    print(f'  {name}: 배경 {mad(filt[:, CH.index(name)]):.1f}')
print('  → 수십이면 µV 결론 강화, 100 이상이면 단위 재의심.\n')

# ③ CAR 전후 배경 — 귓불 기준 공통성분이 배경을 부풀렸는지
common = filt.mean(axis=1, keepdims=True)     # 공통 성분 (samples, 1)
car = filt - common                            # 재참조된 신호
print('=== ③ CAR 전후 배경 ===')
print(f'공통 성분(19채널 평균) 배경: {mad(common):.1f}')
print('채널별 배경:  재참조 전 → 후  (비율)')
for name in ['O1', 'O2', 'Pz', 'P3', 'P4', 'Fp1', 'Fp2']:
    i = CH.index(name)
    bg_b = mad(filt[:, i])
    bg_a = mad(car[:, i])
    print(f'  {name}: {bg_b:.1f} → {bg_a:.1f}  ({bg_a/bg_b:.2f}배)')
print('  → 공통성분이 크고 CAR 후 후두 배경이 0.3~0.5배로 떨어지면 배경 부풀림은 공통성분 탓.')