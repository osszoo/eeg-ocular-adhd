"""탐색·진단 스크립트: 적재 무결성 점검 + PSD 시각화.
파이프라인 함수(load_subject/to_raw, bandpass/car)는 src/ 에서 import해 재사용한다."""
import sys
from pathlib import Path

# src/ 를 import 경로에 추가 (exploration/ 과 src/ 가 프로젝트 루트에 나란히 있는 구조)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_subject, to_raw   # 로딩은 load_data 단일 출처
from preprocess import bandpass, car          # 확정된 파이프라인 함수 재사용

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 (경고 제거)
plt.rcParams['axes.unicode_minus'] = False       # 음수 축 라벨 깨짐 방지

SFREQ = 128
CH = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
      'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']


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

# ============================================================================
# PSD 시각화 — 아래 두 블록은 '서로 다른 질문'에 답하는 별개의 그림이다.
#   [블록 A] 그룹 PSD : "로우패스 상한을 40Hz로 잡아도 되나?"   → 여러 명, 원본만
#   [블록 B] 단계별 PSD: "전처리 단계가 신호를 의도대로 바꿨나?" → 한 명, 단계별
# 둘 다 '원본 PSD'를 그리지만 대상이 달라(8명 vs 1명) 중복이 아니며,
# 한쪽이 다른 쪽을 대체하지 못한다. 둘 다 유지할 것.
# ============================================================================

# --- [블록 A] 그룹 PSD: H_FREQ(=40) 근거 확인용 -----------------------------
# 목적 : 로우패스 상한 결정 근거. 개체차를 넘어선 '공통 패턴'을 봐야 하므로
#        여러 명(ADHD 4 + Control 4)을 겹쳐 그린다. 신호는 전부 필터 전 원본.
# 판독 : 여러 명에서 공통으로 40Hz 위에 의미있는 신경신호가 없고(→상한 40 타당),
#        50Hz에 라인노이즈 봉우리가 보이면 H_FREQ=40 확정 근거가 된다.
# 주의 : 여기 피험자는 8명. 단계 효과가 아니라 '주파수 상한'을 보는 그림이다.
n_each = 4
groups = {
    'ADHD':    sorted(Path('ADHD_part1').glob('*.mat'))[:n_each],
    'Control': sorted(Path('Control_part1').glob('*.mat'))[:n_each],
}
colors = {'ADHD': 'tab:red', 'Control': 'tab:blue'}

fig, ax = plt.subplots(figsize=(11, 4))
for label, files in groups.items():
    for f in files:
        raw = to_raw(load_subject(f))
        psd = raw.compute_psd(fmax=64)
        mean_power = 10 * np.log10(psd.get_data().mean(axis=0))
        ax.plot(psd.freqs, mean_power, color=colors[label],
                alpha=0.6, linewidth=0.8,
                label=label if f == files[0] else None)

ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
       title=f'PSD: ADHD vs Control (각 {n_each}명, 필터 전)')
ax.axvline(40, color='gray', linestyle='--', linewidth=0.8)   # 로우패스 상한 후보
ax.axvline(50, color='black', linestyle=':', linewidth=0.8)   # 전원 라인노이즈
ax.legend(); plt.tight_layout(); plt.show()

# --- [블록 B] 단계별 PSD: 전처리 단계 검증용 --------------------------------
# 목적 : 각 전처리 단계(원본→bandpass→+CAR)가 신호를 의도대로 바꿨는지 확인.
#        단계 '효과'를 보려면 피험자를 고정해야 한다(사람이 바뀌면 변화가
#        전처리 때문인지 개체차 때문인지 구분 불가). 그래서 동일 1명(v10p)만 사용.
# 판독 : ① 원본은 40Hz 위까지 파워가 이어지고 50Hz 라인노이즈가 보인다.
#        ② bandpass 후 40Hz 위가 깎이고 0.5Hz 미만 표류가 정리된다.
#        ③ +CAR 후 모양은 ②와 거의 같되 공통성분만큼 1–30Hz가 몇 dB 내려간다.
#        ④ 셋을 겹쳐 '고주파 절단(①→②)'과 '전체 하강(②→③)'을 한눈에 확인.
# 주의 : 블록 A와 달리 여기 피험자는 1명. 일반화 근거가 아니라 단계 검증용이다.
f = sorted(Path('ADHD_part1').glob('*.mat'))[0]   # 단계 비교용 동일 피험자(v10p)
raw = to_raw(load_subject(f))     # 원본 (필터 전)
filt = bandpass(raw)              # 1단계: 밴드패스
reref = car(filt)                 # 2단계: CAR

# 세 단계 PSD를 미리 계산 (모두 동일 피험자에서 파생 → 직접 비교 가능)
psd_raw  = raw.compute_psd(fmax=64)
psd_filt = filt.compute_psd(fmax=64)
psd_car  = reref.compute_psd(fmax=64)

db = lambda psd: 10 * np.log10(psd.get_data().mean(axis=0))   # 채널평균 → dB

# ① 원본 단독
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(psd_raw.freqs, db(psd_raw), color='tab:blue', linewidth=1.3)
ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
       title='① 원본 PSD (필터 전, v10p)')
plt.tight_layout(); plt.show()

# ② bandpass만
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(psd_filt.freqs, db(psd_filt), color='tab:gray', linewidth=1.3)
ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
       title='② 밴드패스 후 PSD (v10p)')
plt.tight_layout(); plt.show()

# ③ bandpass + CAR
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(psd_car.freqs, db(psd_car), color='tab:green', linewidth=1.3)
ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
       title='③ 밴드패스 + CAR 후 PSD (v10p)')
plt.tight_layout(); plt.show()

# ④ 세 단계 겹쳐 비교
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(psd_raw.freqs,  db(psd_raw),  color='tab:blue',  linewidth=1.3, label='① 원본')
ax.plot(psd_filt.freqs, db(psd_filt), color='tab:gray',  linewidth=1.3, label='② 밴드패스')
ax.plot(psd_car.freqs,  db(psd_car),  color='tab:green', linewidth=1.3, label='③ 밴드패스+CAR')
ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
       title='④ 단계별 PSD 비교 (동일 피험자 v10p)')
ax.legend(); plt.tight_layout(); plt.show()