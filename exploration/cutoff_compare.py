"""ASR cutoff 확정용 비교(진단 전용, 일회성): cutoff off/100/50에서 안구 IC 생존 측정.

목적 : asr_sweep에서 off·100·50으로 좁힌 cutoff 후보 중 하나를 확정한다. 분산이 아니라
       ICA→ICLabel로 '안구 IC가 실제로 살아남는가'를 직접 본다(분산은 안구 손실과
       정상 아티팩트 제거를 구분 못 하므로). 라벨 미사용 → 데이터 누수 방지.

각 피험자 × cutoff(off/100/50)마다 ICA→ICLabel을 돌려 세 지표를 본다.
  - 최대 Eye 확률   : 가장 안구다운 IC의 eye 확률. cutoff 낮출 때 떨어지면 안구 손상.
  - 안구 IC 개수    : eye가 최빈인 IC 수(argmax 기준). 줄면 안구 손상.
  - rank(=n_comp)   : off는 18 기대. 낮은 cutoff에서 줄면 ASR이 차원을 깎는 신호.

피험자는 asr_sweep과 동일(pick_subjects 재사용) — 짧은 녹화 + v265 이상치 포함.
ICA×3cutoff×6명이라 무겁다(피험자당 extended Infomax를 3번). print만 한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from mne_icalabel.iclabel import iclabel_label_components
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인 함수 재사용
from asr_sweep import pick_subjects              # 피험자 선정 로직 재사용(단일 출처)

CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
EYE = CLASSES.index('eye')
CUTOFFS = [None, 100, 50]                         # None = ASR off(기준선)

# 진단 대상: asr_sweep과 동일 피험자 (짧은 녹화 + 전형)
sample = []
sample += [(f, d, 'ADHD')    for f, d in pick_subjects(['ADHD_part1', 'ADHD_part2'])]
sample += [(f, d, 'Control') for f, d in pick_subjects(['Control_part1', 'Control_part2'])]

rows = []
for f, dur, grp in sample:
    cared = car(bandpass(to_raw(load_subject(f))))   # 잠근 순서: bandpass → CAR
    max_eye, n_eye, ranks = [], [], []
    for c in CUTOFFS:
        cleaned = cared if c is None else asr(cared, cutoff=c)   # off=ASR 건너뜀
        comp = ica(cleaned)
        proba = iclabel_label_components(cleaned, comp)          # (n_comp, 7)
        eye = proba[:, EYE]
        max_eye.append(eye.max())
        n_eye.append(int((proba.argmax(axis=1) == EYE).sum()))   # eye가 최빈인 IC 수
        ranks.append(comp.n_components_)
    rows.append({'name': f.stem, 'grp': grp, 'dur': dur,
                 'max_eye': max_eye, 'n_eye': n_eye, 'rank': ranks})


def print_table(title, hint, key, fmt):
    header = f'{"피험자":<8}{"그룹":<8}   ' + '  '.join(
        f'{("off" if c is None else c):>6}' for c in CUTOFFS)
    print(f'\n=== {title} ===')
    print(hint)
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["name"]:<8}{r["grp"]:<8}   '
              + '  '.join(f'{v:>6{fmt}}' for v in r[key]))


print_table('최대 Eye 확률 — 가장 안구다운 IC',
            '  cutoff 낮출 때 떨어지면 안구 IC 손상. 평탄하면 ASR이 안구를 안 건드림.',
            'max_eye', '.2f')
print_table('안구 IC 개수 — eye가 최빈인 IC 수',
            '  cutoff 낮출 때 줄면 안구 손상.', 'n_eye', 'd')
print_table('rank(=n_components) — off는 18 기대',
            '  낮은 cutoff에서 줄면 ASR이 차원을 깎는 신호(안구 섞임 위험).', 'rank', 'd')