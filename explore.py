import scipy.io as sio
import numpy as np
from pathlib import Path

SFREQ = 128

def load_subject(mat_path):
    mat = sio.loadmat(mat_path)
    var = [k for k in mat if not k.startswith('__')][0]
    arr = np.asarray(mat[var], dtype=float)
    if arr.shape[0] == 19 and arr.shape[1] != 19:
        arr = arr.T
    return arr

def load_group(folders, label):
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
    d = [v['dur'] for v in group]
    print(f'[{name}] 길이(초)  min {min(d):.0f} / median {np.median(d):.0f} '
          f'/ mean {np.mean(d):.0f} / max {max(d):.0f}')

stat(adhd, 'ADHD   ')
stat(ctrl, 'Control')

# ICA 데이터 충분성: 임계 미만 피험자 수
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

CH = ['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3',
      'F4','F7','F8','P3','P4','T5','T6','O1','O2']

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