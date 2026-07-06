"""안구 IC 현황 조사(진단 전용, 일회성): 121명에 쓸 만한 안구 IC가 얼마나 있나.

배경 : 안구 특징을 만들려면 먼저 "안구 IC를 가진 피험자가 충분한가"를 알아야 한다.
       rate·파워비율 검증이 6명에서 흔들린 근본 원인이 이걸 몰라서였다.

세 질문 :
  1. 피험자마다 eye≥0.5(ocular 기준)인 IC가 몇 개인가 — 0/1/2+ 분포
  2. eye≥0.5인 IC가 아예 없는 피험자는 몇 명인가 — 많으면 ocular 조건이 흔들림
  3. 그 분포가 ADHD/Control에서 다른가 — 현황 파악용(특징 결정엔 쓰지 않음, 누수 경계)

캐시 재활용 : iclabel_confidence가 _cache/{이름}.npz에 확률행렬(키 'proba')을 저장해둠.
       ICA 재계산 없이 즉시 집계. 라벨은 캐시에 없으므로(누수 방지로 저장 안 함)
       all_subjects()로 파일명↔라벨 매칭.

주의 : '안구 IC 개수'는 데이터 현황이지 특징이 아니다. 기술 통계로만 보고, 이 값을
       보고 특징을 고르지 않는다(그러면 라벨 보고 특징 고르는 누수가 됨).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from load_data import all_subjects

CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')
TAU = 0.5
_CACHE = Path(__file__).resolve().parent / '_cache'

# 라벨 매칭: 파일 stem → 라벨(1=ADHD, 0=Control)
label_of = {f.stem: lab for f, lab in all_subjects()}

rows = []          # (이름, 그룹, eye≥τ IC 개수, 최대 eye 확률)
missing = []       # 캐시 없는 피험자
for name, lab in label_of.items():
    cache_file = _CACHE / f'{name}.npz'
    if not cache_file.exists():
        missing.append(name)
        continue
    proba = np.load(cache_file)['proba']       # (n_comp, 7)
    eye = proba[:, EYE]
    is_eye_argmax = proba.argmax(axis=1) == EYE
    n_ocular = int((is_eye_argmax & (eye >= TAU)).sum())   # eye argmax & ≥τ
    grp = 'ADHD' if lab == 1 else 'Control'
    rows.append({'name': name, 'grp': grp, 'n_ocular': n_ocular, 'max_eye': eye.max()})

if missing:
    print(f'캐시 없는 피험자 {len(missing)}명: {missing[:10]}{" ..." if len(missing)>10 else ""}')
    print('  (iclabel_confidence를 먼저 돌려 캐시를 채워야 전원 집계됨)\n')

n_total = len(rows)
print(f'=== 안구 IC 현황 (ocular 기준: eye argmax & eye≥{TAU}) ===')
print(f'집계 대상: {n_total}명\n')

# --- 질문 1&2: eye≥τ IC 개수 분포 (전체) ---
counts = np.array([r['n_ocular'] for r in rows])
print('[개수 분포 — 전체]')
print(f'  0개 : {int((counts==0).sum()):3d}명   ← ocular 조건 불가')
print(f'  1개 : {int((counts==1).sum()):3d}명')
print(f'  2개+: {int((counts>=2).sum()):3d}명')
print(f'  → 안구 IC 보유(≥1개): {int((counts>=1).sum())}명 / {n_total}명 '
      f'({(counts>=1).mean()*100:.0f}%)')

# --- 질문 3: 그룹별 분포 (현황 파악용) ---
print('\n[개수 분포 — 그룹별 (현황 파악용, 특징 결정엔 미사용)]')
print(f'{"그룹":<9}{"인원":>5}{"0개":>6}{"1개":>6}{"2개+":>6}{"보유율":>8}{"평균개수":>9}')
print('-' * 50)
for grp in ['ADHD', 'Control']:
    g = counts[[i for i, r in enumerate(rows) if r['grp'] == grp]]
    if len(g):
        print(f'{grp:<9}{len(g):>5}{int((g==0).sum()):>6}{int((g==1).sum()):>6}'
              f'{int((g>=2).sum()):>6}{(g>=1).mean()*100:>7.0f}%{g.mean():>9.2f}')

# --- 참고: eye≥τ가 없는 피험자도 최대 eye 확률은 얼마인지 (경계 근처 파악) ---
no_ocular = [r for r in rows if r['n_ocular'] == 0]
if no_ocular:
    near = [r['max_eye'] for r in no_ocular]
    print(f'\n[ocular 0개인 {len(no_ocular)}명의 최대 eye 확률]')
    print(f'  중앙값 {np.median(near):.2f}, 최대 {max(near):.2f} '
          f'— 0.4~0.5 근처가 많으면 τ에 예민, 낮으면 확실히 안구 없음')
    print(f'  0.4~0.5 구간: {sum(0.4<=x<0.5 for x in near)}명, '
          f'0.3 미만: {sum(x<0.3 for x in near)}명')
    
# --- 대표 선정용: 안구 IC 보유 84명 명단 (ID·집단·top-1 eye 확률·녹화 길이) ---
# 에폭 길이·추정기 스윕의 대표는 84명 안에서 골라야 하므로 명단을 출력한다.
# max_eye = 그 피험자에서 eye 확률이 가장 높은 IC 값 → 보유자에겐 top-1 안구 IC 확률.
# 길이는 load_subject로 배열 shape만 읽어 계산(몽타주 불필요, 가벼움).
from load_data import load_subject, SFREQ

path_of = {f.stem: f for f, _ in all_subjects()}    # 이름 → .mat 경로

holders = [r for r in rows if r['n_ocular'] >= 1]
for r in holders:                                    # 길이(초) 채우기
    arr = load_subject(path_of[r['name']])
    r['dur'] = arr.shape[0] / SFREQ
holders.sort(key=lambda r: (r['grp'], -r['max_eye']))  # 집단별, 확률 높은 순

print(f'\n[안구 IC 보유 {len(holders)}명 — 대표 선정용 명단]')
print(f'{"이름":<8}{"그룹":<9}{"안구IC":>6}{"top-1 eye":>11}{"길이(s)":>9}')
print('-' * 43)
for r in holders:
    print(f'{r["name"]:<8}{r["grp"]:<9}{r["n_ocular"]:>6}'
          f'{r["max_eye"]:>11.2f}{r["dur"]:>9.1f}')