"""채널 순서 역추적: 후보 순서별로 지표를 두피에 그려 생리적 타당성 확인.
- 저주파(안구)는 전두에, 알파는 후두에 떠야 '맞는' 순서다."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from load_data import load_subject

mne.set_log_level('WARNING')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
SFREQ = 128

ORDER_TABLE = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
               'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
ORDER_ABSTRACT = ['Fz','Cz','Pz','C3','T7','C4','T8','Fp1','Fp2','F3',
                  'F4','F7','F8','P3','P4','P7','P8','O1','O2']

# 컬럼별 지표 평균 (전 피험자)
files = []
for folder in ['ADHD_part1', 'ADHD_part2', 'Control_part1', 'Control_part2']:
    files += sorted(Path(folder).glob('*.mat'))

low_acc, alpha_acc = [], []
for f in files:
    x = load_subject(f)
    x = x - x.mean(0)
    fr, pxx = signal.welch(x, fs=SFREQ, nperseg=min(512, x.shape[0]), axis=0)
    band = lambda lo, hi: pxx[(fr >= lo) & (fr < hi)].sum(0)
    total = band(0.5, 40) + 1e-12
    low_acc.append(band(0.5, 4) / total)
    alpha_acc.append(band(8, 12) / total)
low   = np.mean(low_acc, 0)      # 컬럼 인덱스 0~18 기준
alpha = np.mean(alpha_acc, 0)

def topo(order, vals, title, ax):
    info = mne.create_info(order, SFREQ, 'eeg')
    info.set_montage('standard_1020')
    mne.viz.plot_topomap(vals, info, axes=ax, show=False, contours=4)
    ax.set_title(title, fontsize=9)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
topo(ORDER_TABLE,    low,   '표순서 · 저주파(전두여야)', axes[0,0])
topo(ORDER_TABLE,    alpha, '표순서 · 알파(후두여야)',   axes[0,1])
topo(ORDER_ABSTRACT, low,   'abstract · 저주파(전두여야)', axes[1,0])
topo(ORDER_ABSTRACT, alpha, 'abstract · 알파(후두여야)',   axes[1,1])
plt.tight_layout(); plt.show()