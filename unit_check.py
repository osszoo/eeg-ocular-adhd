import scipy.io as sio, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path

SFREQ = 128
CH = ['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3',
      'F4','F7','F8','P3','P4','T5','T6','O1','O2']

def load_subject(path):
    mat = sio.loadmat(path)
    var = [k for k in mat if not k.startswith('__')][0]
    arr = np.asarray(mat[var], float)
    if arr.shape[0] == 19 and arr.shape[1] != 19:
        arr = arr.T
    return arr

arr = load_subject(sorted(Path('ADHD_part1').glob('*.mat'))[0])

# 1–40Hz 밴드패스: 표류 + 전원잡음 제거, native 단위 유지
b, a = butter(4, [1/(SFREQ/2), 40/(SFREQ/2)], btype='band')
filt = filtfilt(b, a, arr, axis=0)

# 평균 기준(CAR) 재참조: 각 시점에서 19채널 평균을 빼기
common = filt.mean(axis=1, keepdims=True)     # 공통 성분 (samples, 1)
car = filt - common                            # 재참조된 신호

# 공통 성분 자체의 크기
common_bg = 1.4826 * np.median(np.abs(common - np.median(common)))
print(f'공통 성분(19채널 평균) 배경: {common_bg:.1f}\n')

print('채널별 배경:  재참조 전 → 후  (비율)')
for name in ['O1','O2','Pz','P3','P4','Fp1','Fp2']:
    i = CH.index(name)
    bg_b = 1.4826 * np.median(np.abs(filt[:,i] - np.median(filt[:,i])))
    bg_a = 1.4826 * np.median(np.abs(car[:,i]  - np.median(car[:,i])))
    print(f'  {name}: {bg_b:.1f} → {bg_a:.1f}  ({bg_a/bg_b:.2f}배)')