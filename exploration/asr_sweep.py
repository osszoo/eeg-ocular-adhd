"""ASR cutoff 민감도 스윕(진단 전용): cutoff별 재구성률·전두 분산 보존율 측정.

목적 : ASR cutoff를 확정하기 위한 일회성 탐색. 분류 정확도가 아니라 신호 수준
       지표로만 판단한다(라벨을 보지 않음 → 데이터 누수 방지).

지표 두 가지를 cutoff(off·100·50·30·20)별로 본다.
  - 재구성률(%)        : ASR이 타임라인의 몇 %를 의미있게 바꿨나.
                         0이면 사실상 아무것도 안 함, 과하게 높으면 정상 신호까지 손댐.
  - 전두 분산 보존율(%) : Fp1·Fp2·F7·F8 분산이 ASR 후 얼마나 남나(안구 보존 프록시).
                         cutoff를 낮출수록 떨어지며, 급격히 무너지기 직전이 보존 하한.

적용 순서는 파이프라인 잠근 순서(bandpass → CAR → ASR) 그대로. 'off'는 ASR 미적용
기준선(bandpass+CAR)이라 정의상 재구성률 0 / 보존율 100.

짧은 녹화(<90s)도 일부 포함해 보정 불안정(재구성률 급등) 여부를 함께 본다.
실행하면 표 두 개를 print만 한다(그림 저장 없음).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr     # 잠근 파이프라인 함수 재사용(단일 출처)

SFREQ = 128
FRONTAL = ['Fp1', 'Fp2', 'F7', 'F8']     # 안구(깜빡임·수평 단속운동) 채널
CUTOFFS = [None, 100, 50, 30, 20]        # None = ASR off(기준선)
SHORT_THR = 90                            # 짧은 녹화 기준(초)


def pick_subjects(group_folders, n_typical=2):
    """그룹 폴더에서 진단용 피험자를 고른다: 가장 짧은 1명(보정 불안정 확인용)
    + 중앙값 부근 n_typical명(전형값). (file, dur) 리스트 반환."""
    files = []
    for folder in group_folders:
        for f in sorted(Path(folder).glob('*.mat')):
            dur = load_subject(f).shape[0] / SFREQ
            files.append((f, dur))
    files.sort(key=lambda x: x[1])              # 길이 오름차순
    mid = len(files) // 2
    chosen = [files[0], *files[mid:mid + n_typical]]   # 최단 1 + 중앙값 부근
    # 중복 제거(최단이 중앙값 구간에 겹치는 드문 경우 방지)
    picked, seen = [], set()
    for f, d in chosen:
        if f not in seen:
            picked.append((f, d)); seen.add(f)
    return picked


def recon_fraction(orig, clean, tol=0.05):
    """ASR이 타임라인의 몇 %를 의미있게 바꿨는지(%) — 재구성 범위 프록시.
    채널 SD의 tol배 넘게 값이 바뀐 시점을 '손댄' 것으로 센다."""
    sd = orig.std(axis=1, keepdims=True)
    sd[sd == 0] = 1                                  # 0 division 방지
    touched = (np.abs(clean - orig) > tol * sd).any(axis=0)   # 시점별 변경 여부
    return touched.mean() * 100


def frontal_retention(orig, clean, idx):
    """전두 채널(idx)의 분산이 ASR 후 얼마나 남는지 평균(%).
    100=그대로, 낮을수록 분산(=안구 활동 가능성)이 깎였다는 뜻."""
    v0 = orig[idx].var(axis=1)
    v1 = clean[idx].var(axis=1)
    return float(np.mean(v1 / v0) * 100)


# 진단 대상: ADHD·Control 각 (짧은 1 + 전형 2)명
sample = []
sample += [(f, d, 'ADHD')    for f, d in pick_subjects(['ADHD_part1', 'ADHD_part2'])]
sample += [(f, d, 'Control') for f, d in pick_subjects(['Control_part1', 'Control_part2'])]

# cutoff별 두 지표 수집 (피험자당 ASR을 cutoff 수만큼 돌리므로 다소 무겁다)
rows = []
for f, dur, grp in sample:
    cared = car(bandpass(to_raw(load_subject(f))))   # 잠근 순서: bandpass → CAR
    base = cared.get_data()                          # ASR 기준선(off)
    idx = [cared.ch_names.index(ch) for ch in FRONTAL]

    recon, retain = [], []
    for c in CUTOFFS:
        if c is None:                                # off = 기준선
            recon.append(0.0); retain.append(100.0)
        else:
            clean = asr(cared, cutoff=c).get_data()  # 잠근 asr() 그대로 사용
            recon.append(recon_fraction(base, clean))
            retain.append(frontal_retention(base, clean, idx))
    rows.append({'name': f.stem, 'grp': grp, 'dur': dur, 'recon': recon, 'retain': retain})


def print_table(title, hint, key):
    """한 지표를 피험자(행)×cutoff(열) 표로 출력."""
    header = f'{"피험자":<8}{"그룹":<8}{"길이":>5}   ' + '  '.join(
        f'{("off" if c is None else c):>6}' for c in CUTOFFS)
    print(f'\n=== {title} ===')
    print(hint)
    print(header)
    print('-' * len(header))
    for r in rows:
        flag = '*' if r['dur'] < SHORT_THR else ' '   # 짧은 녹화 표시
        print(f'{r["name"]:<8}{r["grp"]:<8}{r["dur"]:>4.0f}{flag}  '
              + '  '.join(f'{v:>6.1f}' for v in r[key]))
    print('  (* = 90초 미만 짧은 녹화)')


print_table('재구성률(%) — ASR이 타임라인을 손댄 비율',
            '  낮을수록 적게 건드림. 과도하게 높으면 정상 신호까지 제거 의심.', 'recon')
print_table('전두 분산 보존율(%) — Fp1·Fp2·F7·F8',
            '  100=그대로. cutoff 낮출 때 급격히 무너지는 지점 직전이 안구 보존 하한.', 'retain')