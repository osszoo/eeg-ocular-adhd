"""뇌파 TBR 분포 점검(진단 전용, 일회성): 극단 TBR이 소수 outlier인지,
특정 피험자에 몰렸는지 확인. 라벨은 보지 않는다(누수 경계).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import csv
import numpy as np
from collections import Counter

CSV = Path(__file__).resolve().parent.parent / 'results' / 'features_neural.csv'
TBR_COLS = ['frontal_tbr', 'central_tbr', 'parietal_tbr', 'occipital_tbr', 'temporal_tbr']
THRESH = 10.0     # 이 값 넘으면 '극단'으로 셈

# csv 로드
with open(CSV, newline='') as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
print(f'총 에폭(행): {len(rows)}\n')

# 부위별 TBR 분포 요약
print('[부위별 TBR 분포]')
print(f'{"부위":<10}{"중앙값":>8}{"90%":>8}{"99%":>8}{"최대":>8}{">10개":>7}')
print('-' * 49)
for col in TBR_COLS:
    v = np.array([float(r[col]) for r in rows])
    n_ext = int((v > THRESH).sum())
    print(f'{col:<10}{np.median(v):>8.2f}{np.percentile(v,90):>8.2f}'
          f'{np.percentile(v,99):>8.2f}{v.max():>8.2f}{n_ext:>7}')

# 극단 TBR(어느 부위든 THRESH 초과)이 어느 피험자에 몰렸나
ext_subj = Counter()
for r in rows:
    if any(float(r[c]) > THRESH for c in TBR_COLS):
        ext_subj[r['subject_id']] += 1

total_ext = sum(ext_subj.values())
print(f'\n[극단 TBR(>{THRESH:.0f}) 피험자 분포]')
print(f'  극단 에폭이 있는 피험자: {len(ext_subj)}명 / 전체 극단 에폭 {total_ext}개')
print(f'  상위 피험자(극단 에폭 수):')
for name, cnt in ext_subj.most_common(10):
    print(f'    {name:<8} {cnt}개')