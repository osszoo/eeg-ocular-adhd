"""탐색·진단 스크립트: 적재 무결성 점검 + PSD 시각화(화면 표시 전용).

이 파일이 하는 일은 두 가지뿐이다.
  1. 적재된 데이터의 무결성·크기·아티팩트 지표를 print로 점검.
  2. figures.py 가 만든 PSD 그림을 화면에 띄워 눈으로 확인.

그림을 '만드는' 코드는 여기 없다(전부 figures.py). results/ 에 '저장'하는
것도 여기서 하지 않는다(그건 save_figures.py 담당). 즉 이 파일은 탐색 전용이라
마음껏 돌리고 고쳐도 results/ 산출물에는 영향이 없다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_subject                     # 진단 통계에서 사용
from figures import make_group_psd, make_stage_psds    # 그림 생성은 figures.py 재사용
# 한글 폰트 설정은 figures.py 가 import 시점에 처리하므로 여기서 따로 하지 않는다.

SFREQ = 128
CH = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
      'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']


def load_group(folders, label):
    """폴더 목록의 모든 .mat 을 적재해 피험자별 진단 지표를 dict로 모은다.
    (채널수 / 길이 / 진폭 / NaN 여부 / 채널별 표준편차·MAD·첨도·차분MAD)"""
    out = {}
    for folder in folders:
        for f in sorted(Path(folder).glob('*.mat')):
            arr = load_subject(f)
            out[f.stem] = {
                'label': label,
                'n_ch': arr.shape[1],
                'dur': arr.shape[0] / SFREQ,
                'amp_min': arr.min(),
                'amp_max': arr.max(),
                'has_nan': np.isnan(arr).any(),
                'ch_std': arr.std(axis=0),   # 채널별 표준편차 (19개)
                'ch_mad': np.median(np.abs(arr - np.median(arr, axis=0)), axis=0),
                'ch_kurt': ((arr - arr.mean(0))**4).mean(0) / (arr.var(0)**2) - 3,
                'ch_dmad': np.median(np.abs(np.diff(arr, axis=0)), axis=0),
            }
    return out


subjects = {}
subjects.update(load_group(['ADHD_part1', 'ADHD_part2'], 1))
subjects.update(load_group(['Control_part1', 'Control_part2'], 0))

adhd = [v for v in subjects.values() if v['label'] == 1]
ctrl = [v for v in subjects.values() if v['label'] == 0]
print(f'총 {len(subjects)}명 (ADHD {len(adhd)}, Control {len(ctrl)})\n')

def stat(group, name):
    """그룹의 기록 길이(초) 분포를 min/median/mean/max 로 요약 출력."""
    d = [v['dur'] for v in group]
    print(f'[{name}] 길이(초)  min {min(d):.0f} / median {np.median(d):.0f} '
          f'/ mean {np.mean(d):.0f} / max {max(d):.0f}')

stat(adhd, 'ADHD   ')
stat(ctrl, 'Control')

# ICA 데이터 충분성 사전 점검 (ICA를 돌리는 게 아니라, 길이가 짧아 나중에
# 성분 분리가 불안정할 수 있는 피험자가 몇 명인지 미리 세어두는 카운트)
for thr in (60, 90):
    n = sum(v['dur'] < thr for v in subjects.values())
    print(f'  {thr}초 미만 피험자: {n}명')

# 이상 징후 점검
bad_ch = [k for k, v in subjects.items() if v['n_ch'] != 19]
nan_s  = [k for k, v in subjects.items() if v['has_nan']]
amp = np.array([[v['amp_min'], v['amp_max']] for v in subjects.values()])
print(f'\n채널수≠19 : {bad_ch if bad_ch else "없음"}')
print(f'NaN 포함  : {nan_s if nan_s else "없음"}')
print(f'전체 진폭 범위: {amp[:,0].min():.1f} ~ {amp[:,1].max():.1f}')

std_med  = np.median(np.vstack([v['ch_std']  for v in subjects.values()]))
mad_med  = np.median(np.vstack([v['ch_mad']  for v in subjects.values()]))
print(f'\n전형적 크기:  STD 중앙값 {std_med:.1f}  |  MAD 중앙값 {mad_med:.1f}')
print('  → MAD가 10~40이면 정상 µV(아티팩트가 STD를 키운 것),'
      ' 100 이상이면 단위 의심')

kurt = np.vstack([v['ch_kurt'] for v in subjects.values()]).mean(axis=0)
order = kurt.argsort()[::-1]
print('\n첨도 큰 채널 top5 (안구 이벤트 지표):',
      [(CH[i], round(kurt[i], 1)) for i in order[:5]])
print('  → Fp1·Fp2·F7·F8가 올라오면 안구 신호가 살아 있다는 증거')

dmad_med = np.median(np.vstack([v['ch_dmad'] for v in subjects.values()]))
print(f'\n차분 기반 크기(표류 제거): diff-MAD 중앙값 {dmad_med:.1f}')
print('  → 수십이면 정상 µV(표류가 MAD를 키운 것), 수백이면 단위가 µV 아님(정수 스케일 의심)')

# ── PSD 시각화 (화면 표시 전용) ─────────────────────────────────────────────
# 그림은 figures.py 가 만들고, 여기서는 띄우기만 한다. results/ 에 남기려면
# 이 파일이 아니라 save_figures.py 를 실행할 것.
#   make_group_psd()  : [블록 A] 그룹 PSD     — 로우패스 상한(40Hz) 근거
#   make_stage_psds() : [블록 B] 단계별 PSD ①②③④ — 전처리 단계 검증
# 반환 figure를 변수로 받지 않아도 된다. matplotlib이 생성된 figure를 내부적으로
# 들고 있어, 마지막 plt.show() 한 번으로 위에서 만든 그림이 전부 화면에 뜬다.
make_group_psd()    # 블록 A
make_stage_psds()   # 블록 B
plt.show()