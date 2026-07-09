"""combined vs neural84 시드 반복(진단 전용, 일회성).

목적 : classify.py 결과에서 RandomForest만 ΔAUC가 음수로 어긋났는데, 이게
       단일 random_state(42)의 fold 변동(우연)인지 구조적인지 확인한다.
       여러 시드로 반복해 Δ정확도·ΔAUC가 시드에 걸쳐 안정적인지 본다.

방법 : random_state를 10개 바꿔가며(shuffle=True) fold 분할·모델 초기화를 흔든다.
       시드마다 combined84 − neural84의 Δ정확도·ΔAUC를 3분류기별로 모아,
       평균·범위·양수 개수로 판단.

정직성 : '제일 좋은 시드 고르기'는 하지 않는다(체리피킹 금지). 10개 전부를 분포로 본다.
누수 경계 : 시드는 CV 분할·모델 초기화만 바꾼다. 라벨·특징 불변.

classify.py 재사용 : load_xy·evaluate·CONDITIONS·K를 그대로 import(단일 출처).
       단, classify.make_classifiers는 시드가 42로 고정이라, 시드 받는 팩토리는
       여기서 별도 정의(설정값은 classify.py와 동일).
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')       # sklearn deprecation 경고 숨김(결과 무관)

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from classify import load_xy, evaluate, CONDITIONS, K   # 단일 출처 재사용

SEEDS = list(range(10))
NAMES = ['RandomForest', 'Logistic', 'SVM-RBF']


def factories(seed):
    """시드 받는 분류기 팩토리. 설정값은 classify.py와 동일, random_state만 가변."""
    return {
        'RandomForest': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                random_state=seed))]),
        'Logistic': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=seed))]),
        'SVM-RBF': lambda: Pipeline([
            ('sc', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        probability=True, random_state=seed))]),
    }


if __name__ == '__main__':
    Xn, yn, gn, _ = load_xy(CONDITIONS['neural84'])
    Xc, yc, gc, _ = load_xy(CONDITIONS['combined84'])
    assert np.array_equal(gn, gc) and np.array_equal(yn, yc), \
        'neural84/combined84 사람·라벨 불일치 — 조립 확인'

    dacc = {n: [] for n in NAMES}
    dauc = {n: [] for n in NAMES}

    print(f'=== combined84 vs neural84 시드 반복 ({len(SEEDS)}개) ===')
    print('시드마다 fold 분할·모델 초기화가 달라짐. 최고시드 고르기 없음.\n')

    for si, seed in enumerate(SEEDS, 1):
        sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(yn)), yn, gn))   # 두 조건 같은 분할 공유
        fac = factories(seed)
        for n in NAMES:
            mn = evaluate(Xn, yn, gn, splits, fac[n])
            mc = evaluate(Xc, yc, gc, splits, fac[n])
            dacc[n].append(float((mc['acc'] - mn['acc']).mean()))
            dauc[n].append(float(np.nanmean(mc['auc'] - mn['auc'])))
        print(f'  시드 {si}/{len(SEEDS)} 완료')

    print(f'\n{"분류기":<14}{"Δacc평균":>10}{"[최소~최대]":>18}{"양수":>7}'
          f'{"ΔAUC평균":>10}{"[최소~최대]":>18}{"양수":>7}')
    print('-' * 84)
    for n in NAMES:
        a = np.array(dacc[n]); u = np.array(dauc[n])
        print(f'{n:<14}{a.mean():>+10.3f}{f"[{a.min():+.3f}~{a.max():+.3f}]":>18}'
              f'{int((a > 0).sum()):>5}/10'
              f'{u.mean():>+10.3f}{f"[{u.min():+.3f}~{u.max():+.3f}]":>18}'
              f'{int((u > 0).sum()):>5}/10')

    print('\n[판정 가이드]')
    print('  양수 8+/10  → 시드 무관하게 combined>neural (안정적 이득)')
    print('  양수 4~6/10 → fold 변동에 흔들림 (결론 유보)')
    print('  범위가 부호를 넘나들면(예: -0.05~+0.06) → 그 지표는 우연 수준')