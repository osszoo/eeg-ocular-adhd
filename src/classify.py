"""조건별 분류·평가(정식 파이프라인).

assemble.py가 만든 조건별 표를 subject-wise CV로 평가하고, 핵심 비교
(combined > neural84)가 분류기와 무관하게 일관된지 본다.

CV : StratifiedGroupKFold(K=5) — groups=subject_id(사람 단위 분할),
     stratify=label(집단 비율 유지). 같은 사람 에폭이 train/test에 안 섞임(누수 방지).
정규화 : Pipeline[StandardScaler → 분류기]. scaler는 fold의 train에서만 fit되고
     test엔 그 값으로 apply만 됨(누수 구조적 차단).
사람 단위 평가 : 에폭별 예측확률 → subject_id로 평균 → 그 사람의 ADHD 확률.
     임계 0.5로 이진(정확도·민감도·특이도) + 연속 점수로 AUC.

분류기 3개(튜닝 없음, 보수적 고정) : RandomForest · LogisticRegression · SVM-RBF.
     목적은 최고 성능이 아니라 강건성 — 3개 다에서 combined>neural이면 '안구가 돕는다'.
조건 : neural84 · ocular84 · combined84(주) + neural121(보조 참고, 직접 대결 안 함).

비교 : combined84 − neural84를 fold별 짝지어(같은 fold=같은 사람). 5-fold 일관성 +
     3분류기 일관성으로 논증(84명·5-fold엔 p값보다 패턴이 정직).

누수 경계 : 라벨은 학습에만 쓰고 특징·정규화엔 안 씀. 하이퍼파라미터 튜닝 안 함
     (튜닝하면 nested CV 필요). 84 조건 3개는 같은 fold를 공유해 짝비교가 성립.
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
import sys
import csv
from pathlib import Path
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# stdout을 UTF-8로 고정 — 파이프/리다이렉트 시 cp949 인코딩 크래시 방지(계측)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / 'results'
K = 5
RANDOM_STATE = 42

CONDITIONS = {                      # 조건 → 표 파일
    'neural84':   _RESULTS / 'X_neural84.csv',
    'ocular84':   _RESULTS / 'X_ocular84.csv',
    'combined84': _RESULTS / 'X_combined84.csv',
    'neural121':  _RESULTS / 'X_neural121.csv',
}


# ── 분류기 팩토리(매 fold 새로 생성; Pipeline으로 정규화 결합) ──
def make_classifiers():
    """이름 → Pipeline 생성 함수. 보수적 고정값, 튜닝 없음."""
    return {
        'RandomForest': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                random_state=RANDOM_STATE, class_weight='balanced'))]),
        'Logistic': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2', C=1.0, max_iter=1000, random_state=RANDOM_STATE,
                class_weight='balanced'))]),
        'SVM-RBF': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        probability=True, random_state=RANDOM_STATE,
                        class_weight='balanced'))]),
    }


# ── 로드: csv → X, y, groups ────────────────────────────────
def load_xy(path):
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    groups = np.array([row[0] for row in rows])            # subject_id
    y = np.array([int(row[1]) for row in rows])            # label
    X = np.array([[float(v) for v in row[2:]] for row in rows])
    return X, y, groups, header[2:]


# ── 사람 단위 집계: 에폭 확률 → subject 평균 ────────────────
def subject_scores(subj_ids, y_true, proba):
    """에폭별 P(ADHD)를 사람별 평균. 반환 (사람 실제라벨, 사람 평균확률)."""
    agg = OrderedDict()
    for sid, yt, p in zip(subj_ids, y_true, proba):
        if sid not in agg:
            agg[sid] = {'y': int(yt), 'ps': []}
        agg[sid]['ps'].append(p)
    subs = list(agg.keys())
    st = np.array([agg[s]['y'] for s in subs])
    sp = np.array([np.mean(agg[s]['ps']) for s in subs])
    return st, sp


def fold_metrics(st, sp):
    """사람 단위 (실제, 평균확률) → 정확도·AUC·민감도·특이도."""
    pred = (sp >= 0.5).astype(int)
    acc = float((pred == st).mean())
    auc = float(roc_auc_score(st, sp)) if len(set(st)) == 2 else np.nan
    tp = int(((pred == 1) & (st == 1)).sum()); fn = int(((pred == 0) & (st == 1)).sum())
    tn = int(((pred == 0) & (st == 0)).sum()); fp = int(((pred == 1) & (st == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan     # ADHD 검출
    spec = tn / (tn + fp) if (tn + fp) else np.nan     # Control 검출
    return acc, auc, sens, spec


# ── 한 조건 평가: fold별 사람 단위 지표 ─────────────────────
def evaluate(X, y, groups, splits, make_clf):
    out = {'acc': [], 'auc': [], 'sens': [], 'spec': []}
    for tr, te in splits:
        clf = make_clf()
        clf.fit(X[tr], y[tr])                            # scaler·분류기 train서만 fit
        proba = clf.predict_proba(X[te])[:, 1]           # P(ADHD=1) 에폭별
        st, sp = subject_scores(groups[te], y[te], proba)
        a, u, se, sp_ = fold_metrics(st, sp)
        out['acc'].append(a); out['auc'].append(u)
        out['sens'].append(se); out['spec'].append(sp_)
    return {k: np.array(v) for k, v in out.items()}


def ms(a):
    """평균±표준편차 문자열(NaN 무시)."""
    return f'{np.nanmean(a):.3f}±{np.nanstd(a):.3f}'


class _Tee:
    """print 출력을 콘솔과 파일에 동시에 흘려보냄(결과 저장·계측)."""
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


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 결과를 콘솔에 띄우면서 동시에 파일로 저장(교정 결과 기록)
    _OUT_DIR = _RESULTS / 'corrected'
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _out_fh = open(_OUT_DIR / 'classify_corrected.txt', 'w', encoding='utf-8')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _out_fh)

    data = {c: load_xy(p) for c, p in CONDITIONS.items()}

    # 84 조건 3개는 같은 사람·같은 순서 → 같은 fold 공유(짝비교 성립) 확인
    g_ref, y_ref = data['neural84'][2], data['neural84'][1]
    for c in ('ocular84', 'combined84'):
        assert np.array_equal(data[c][2], g_ref) and np.array_equal(data[c][1], y_ref), \
            f'{c}의 subject_id/label이 neural84와 불일치 — 조립 정렬 확인 필요'

    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    splits84 = list(sgkf.split(np.zeros(len(y_ref)), y_ref, g_ref))
    X121, y121, g121, _ = data['neural121']
    splits121 = list(sgkf.split(np.zeros(len(y121)), y121, g121))

    clfs = make_classifiers()
    results = {}       # (조건, 분류기) → 지표 dict

    print('=== subject-wise CV 분류 (StratifiedGroupKFold K=5, 사람 단위 평가) ===\n')
    for cond in CONDITIONS:
        X, y, groups, _ = data[cond]
        splits = splits121 if cond == 'neural121' else splits84
        tag = ' (보조·참고)' if cond == 'neural121' else ''
        print(f'[{cond}]{tag}  {len(set(groups))}명 · {len(y)}에폭')
        print(f'  {"분류기":<14}{"정확도":>12}{"AUC":>12}{"민감도":>12}{"특이도":>12}')
        for name, make in clfs.items():
            m = evaluate(X, y, groups, splits, make)
            results[(cond, name)] = m
            print(f'  {name:<14}{ms(m["acc"]):>12}{ms(m["auc"]):>12}'
                  f'{ms(m["sens"]):>12}{ms(m["spec"]):>12}')
        print()

    # ── 핵심 비교: combined84 − neural84 (fold별 짝, 분류기마다) ──
    print('=== 핵심 비교: combined84 vs neural84 (fold별 짝지음) ===')
    print(f'  {"분류기":<14}{"Δ정확도(평균)":>16}{"양수 fold":>11}{"ΔAUC(평균)":>14}{"양수 fold":>11}')
    verdict = []
    for name in clfs:
        dn_acc = results[('combined84', name)]['acc'] - results[('neural84', name)]['acc']
        dn_auc = results[('combined84', name)]['auc'] - results[('neural84', name)]['auc']
        pos_acc = int((dn_acc > 0).sum())
        pos_auc = int((dn_auc > 0).sum())
        print(f'  {name:<14}{dn_acc.mean():>+16.3f}{pos_acc:>8}/{K}'
              f'{np.nanmean(dn_auc):>+14.3f}{pos_auc:>8}/{K}')
        verdict.append(dn_acc.mean() > 0 and np.nanmean(dn_auc) > 0)

    print('\n[강건성 판정]')
    if all(verdict):
        print('  ✓ 3개 분류기 모두에서 combined > neural84 (정확도·AUC 평균) '
              '— 안구 기여가 분류기와 무관하게 일관됨.')
    else:
        n_pos = sum(verdict)
        print(f'  △ {n_pos}/3 분류기에서만 combined > neural84 — 일관성 부족. '
              '체리피킹 없이 정직하게 보고할 것(안구가 뇌파와 중복이거나 약함).')
    print('\n  * neural121은 84명과 사람이 달라 직접 대결 대상 아님(참고용 baseline).')
    print('  * 84명·5-fold라 p값보다 일관성 패턴으로 논증(설계 확정).')

    sys.stdout = _orig_stdout
    _out_fh.close()