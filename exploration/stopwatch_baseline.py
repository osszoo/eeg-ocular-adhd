"""초시계 기준선(stopwatch baseline) 진단 — 4단계.

가설과 무관한 교란 측정: 녹화 길이(초)만으로 ADHD/대조군을 얼마나 가르는가.
자기진행 과제라 길이 = 총 소요시간 → ADHD가 더 김(확립된 소견). 길이 단독 AUC가
combined 파이프라인 AUC와 맞먹으면, 파이프라인 성능이 신경/안구가 아니라 길이 누수일
수 있음을 뜻함 → 반드시 보고할 기준선(동시에 방법론 기여).

두 값을 뽑는다:
  ① 순수 AUC : roc_auc(길이, 라벨) — 학습·fold 없이 가장 정직한 초시계.
  ② CV  AUC : classify.py와 '완전히 같은 fold'로 test fold마다 (라벨, 길이) AUC → 평균.
              같은 fold를 쓰려고 X_*.csv의 에폭 단위 subject_id/label로 동일 분할 재현.
84명(안구 IC 보유)·121명(전체) 각각. ADHD/대조군 길이 중앙값도 출력(교란 실재 확인).

길이는 .mat 원본 샘플 수 ÷ SFREQ(전처리 전 총 녹화시간). load_subject는 (samples,19).
"""
import sys
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from load_data import all_subjects, load_subject, SFREQ   # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_RESULTS = _ROOT / 'results'
K = 5
RANDOM_STATE = 42


def lengths_by_subject():
    """{name: (label, length_sec)} — .mat 원본 길이
전처리 전, 총 과제시간)."""
    out = {}
    for f, label in all_subjects():
        n_samples = load_subject(f).shape[0]
        out[f.stem] = (int(label), n_samples / SFREQ)
    return out


def epoch_level(csv_path):
    """X_*.csv → (groups_epoch, y_epoch) 에폭 단위 — classify.py와 동일 분할 재현용."""
    with open(csv_path, newline='') as fh:
        r = csv.reader(fh)
        next(r)
        rows = [(row[0], int(row[1])) for row in r]
    groups = [g for g, _ in rows]
    y = np.array([lab for _, lab in rows])
    return groups, y


def stopwatch(csv_path, tab):
    """조건(84 또는 121)의 순수 AUC·CV AUC·(사람 라벨, 사람 길이) 반환."""
    g_ep, y_ep = epoch_level(csv_path)
    subs = list(dict.fromkeys(g_ep))                 # 등장 순서 유지, 중복 제거
    y_sub = np.array([tab[s][0] for s in subs])
    len_sub = np.array([tab[s][1] for s in subs])

    # ① 순수: 사람 단위 길이→라벨 직접 AUC(학습·fold 없음)
    auc_pure = roc_auc_score(y_sub, len_sub)

    # ② CV: classify.py와 동일한 에폭 단위 분할 → test 피험자의 길이 AUC 평균
    sub_index = {s: i for i, s in enumerate(subs)}
    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    for _, te in sgkf.split(np.zeros(len(y_ep)), y_ep, g_ep):
        te_subs = list(dict.fromkeys(g_ep[i] for i in te))   # 이 fold의 test 피험자
        idx = [sub_index[s] for s in te_subs]
        yy, ll = y_sub[idx], len_sub[idx]
        if len(set(yy.tolist())) == 2:
            fold_aucs.append(roc_auc_score(yy, ll))
    auc_cv = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
    return auc_pure, auc_cv, y_sub, len_sub


class _Tee:
    """콘솔+파일 동시 출력."""
    encoding = 'utf-8'

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except (ValueError, OSError):
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except (ValueError, OSError):
                pass


if __name__ == '__main__':
    out_dir = _RESULTS / 'corrected'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / 'stopwatch_baseline.txt', 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)

    tab = lengths_by_subject()

    print('=== 초시계 기준선 (녹화 길이 단독) ===\n')
    print(f'  {"조건":<8}{"순수AUC":>10}{"CV AUC":>10}'
          f'{"ADHD중앙(s)":>14}{"Ctrl중앙(s)":>14}{"Δ중앙":>10}')
    for name, path in (('84명', _RESULTS / 'X_neural84.csv'),
                       ('121명', _RESULTS / 'X_neural121.csv')):
        ap, ac, ys, ls = stopwatch(path, tab)
        ma = float(np.median(ls[ys == 1]))
        mc = float(np.median(ls[ys == 0]))
        print(f'  {name:<8}{ap:>10.3f}{ac:>10.3f}'
              f'{ma:>14.1f}{mc:>14.1f}{ma - mc:>+10.1f}')

    print('\n  * 순수 = roc_auc(길이, 라벨), fold 없음. '
          'CV = classify.py와 동일 분할로 test fold별 AUC 평균.')
    print('  * 이 AUC가 combined 파이프라인 AUC와 맞먹으면 길이 교란이 실재 '
          '— 5단계에서 통제.')

    sys.stdout = orig
    fh.close()
