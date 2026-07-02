"""ICLabel 확신도 분포 진단(진단 전용, 일회성): argmax vs 임계값(τ) 결정 근거.

목적 : IC를 neural/ocular로 가르는 기준을 정하기 전에, 121명 전체 IC의 ICLabel
       확신도가 실제로 어떻게 분포하는지 본다. 핵심 질문 —
       "argmax가 '저확신'으로 eye에 억지 배정하는 IC가 몇 %냐?"
       거의 없으면 argmax와 τ는 실무적으로 동일 → 단순한 argmax 채택.
       많으면 τ(임계값)로 ocular 풀을 깨끗이 거를 값어치가 생김.

       라벨(ADHD/Control)·분류 정확도를 일절 안 본다 → 데이터 누수 아님.
       신호 수준 진단만 한다(확률 분포 히스토그램).

파이프라인은 잠근 최종본 그대로: bandpass → CAR → ASR(cutoff=100) → ICA → ICLabel.
피험자 열거는 load_data.all_subjects()(단일 출처). cutoff는 preprocess.ASR_CUTOFF 사용.

IC별 세 지표:
  - top class : ICLabel 7클래스 중 argmax(최빈 클래스)
  - top prob  : 그 argmax 클래스의 확률(= 확신도). 낮으면 argmax가 억지 배정한 것.
  - margin    : top - second. 작으면 top prob이 높아도 두 클래스가 팽팽(애매).

무겁다: 121명 × ICA(extended Infomax) 1회. 첫 실행은 수십 분.
확률 행렬을 피험자별 npz로 캐시 → 두 번째부터는 집계·플롯만 즉시 돈다.
파이프라인을 고쳤으면 FORCE=True로 캐시를 무시하고 전원 재계산할 것.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from mne_icalabel.iclabel import iclabel_label_components
from load_data import all_subjects, load_subject, to_raw
from preprocess import bandpass, car, asr, ica, ASR_CUTOFF   # 잠근 파이프라인 재사용

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 (경고 제거)
plt.rcParams['axes.unicode_minus'] = False

CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')

# 캐시: 확률 행렬 원본만 저장(해석은 저장 안 함 — 임계값은 바꿔가며 다시 볼 것).
_CACHE = Path(__file__).resolve().parent / '_cache'
_CACHE.mkdir(exist_ok=True)
FORCE = False   # True면 캐시 무시하고 전원 재계산(파이프라인 수정 후엔 켤 것)


def proba_for(mat_path):
    """한 피험자의 ICLabel 확률 행렬 (n_comp, 7)을 반환. 캐시 사용."""
    cache_file = _CACHE / f'{mat_path.stem}.npz'
    if cache_file.exists() and not FORCE:
        return np.load(cache_file)['proba']
    # 잠근 순서: bandpass → CAR → ASR(확정값) → ICA → ICLabel
    cleaned = asr(car(bandpass(to_raw(load_subject(mat_path)))), cutoff=ASR_CUTOFF)
    comp = ica(cleaned)
    proba = iclabel_label_components(cleaned, comp)   # (n_comp, 7)
    np.savez(cache_file, proba=proba)
    return proba


# --- 전원 순회: IC별 (top class, top prob, margin) 수집 ---
subjects = all_subjects()
top_cls, top_prob, margin = [], [], []
for i, (f, _label) in enumerate(subjects, 1):   # 라벨은 안 씀(누수 방지)
    proba = proba_for(f)
    order = np.sort(proba, axis=1)               # 행별 오름차순
    top_cls.extend(proba.argmax(axis=1).tolist())
    top_prob.extend(order[:, -1].tolist())       # 최댓값
    margin.extend((order[:, -1] - order[:, -2]).tolist())   # 최댓값 - 두 번째
    print(f'[{i:3d}/{len(subjects)}] {f.stem:6s} ICs={proba.shape[0]}')

top_cls = np.array(top_cls)
top_prob = np.array(top_prob)
margin = np.array(margin)
print(f'\n총 IC 수: {len(top_cls)}개 (피험자 {len(subjects)}명)')


# --- 표 1: top class별 IC 개수 + 그 확신도 요약 ---
print('\n=== top class 분포 (argmax 기준 전체 IC) ===')
print(f'{"class":<14}{"IC수":>6}{"top prob 중앙값":>16}{"margin 중앙값":>14}')
print('-' * 50)
for k, name in enumerate(CLASSES):
    m = top_cls == k
    if m.any():
        print(f'{name:<14}{m.sum():>6}{np.median(top_prob[m]):>16.2f}'
              f'{np.median(margin[m]):>14.2f}')


# --- 표 2: eye가 argmax인 IC의 확신도 버킷 (핵심) ---
eye_mask = top_cls == EYE
eye_prob = top_prob[eye_mask]
n_eye = len(eye_prob)
print(f'\n=== eye가 argmax인 IC({n_eye}개)의 top prob 버킷 ===')
print('  argmax만 쓰면 이 전체가 ocular 풀에 들어간다.')
print('  <0.5 비중이 크면 저확신 IC가 섞이는 것 → τ로 거를 값어치.')
if n_eye:
    buckets = [('< 0.5', eye_prob < 0.5),
               ('0.5 ~ 0.7', (eye_prob >= 0.5) & (eye_prob < 0.7)),
               ('>= 0.7', eye_prob >= 0.7)]
    for lab, bm in buckets:
        c = int(bm.sum())
        print(f'  {lab:<10} {c:>4}개  ({c / n_eye * 100:4.1f}%)')


# --- 플롯: top prob 히스토그램 (전체 + eye 강조) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.hist(top_prob, bins=20, range=(0, 1), color='tab:gray', edgecolor='white')
ax1.axvline(0.5, color='tab:red', ls='--', lw=1, label='τ=0.5')
ax1.axvline(0.7, color='tab:orange', ls='--', lw=1, label='τ=0.7')
ax1.set(xlabel='top-class 확률 (확신도)', ylabel='IC 수',
        title=f'전체 IC({len(top_prob)}개) 확신도 분포')
ax1.legend()

if n_eye:
    ax2.hist(eye_prob, bins=20, range=(0, 1), color='tab:blue', edgecolor='white')
    ax2.axvline(0.5, color='tab:red', ls='--', lw=1, label='τ=0.5')
    ax2.axvline(0.7, color='tab:orange', ls='--', lw=1, label='τ=0.7')
    ax2.set(xlabel='eye 확률', ylabel='IC 수',
            title=f'eye가 argmax인 IC({n_eye}개)')
    ax2.legend()

plt.tight_layout()
plt.show()