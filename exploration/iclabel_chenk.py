"""ICLabel 검증(진단 전용, 단일 피험자): ICA가 분리한 IC에 ICLabel을 붙여
Eye 확률을 확인한다. 토포로 안구라 추정한 IC가 실제로 Eye 상위인지 수치로 검증.

ICLabel 0.9.0 요구사항과 우리 파이프라인의 정합성:
  - extended infomax 분해   → ica()가 충족 (method='infomax', extended=True)
  - 평균참조                → car()가 충족
  - 1–100Hz 밴드패스 기대   → 우리는 0.5–40Hz (알려진 OOD). 경고가 떠도 에러 아님.
                              Eye는 전두·저주파라 우리 대역에서 상대적으로 견고.

지금은 임계값을 잠그지 않는다. Eye 확률값 자체를 보는 단계(임계값 잠금은 cutoff
확정 후). 출력은 IC별 7클래스 중 Eye 확률을 내림차순으로 표시.
잠근 순서(bandpass → CAR → ASR → ICA) 그대로 단일 피험자(v10p)에 적용.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from mne_icalabel import label_components
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인 함수 재사용(단일 출처)

# ICLabel 7클래스 순서(문서 명시). Eye='eye blink'는 인덱스 2.
CLASSES = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']

SUBJ = next(p for d in ['ADHD_part1', 'ADHD_part2', 'Control_part1', 'Control_part2']
            for p in Path(d).glob('v10p.mat'))

cleaned = asr(car(bandpass(to_raw(load_subject(SUBJ)))))
ic = ica(cleaned)

# ICLabel 실행: IC×7 확률 배열 반환
result = label_components(cleaned, ic, method='iclabel')
proba = result['y_pred_proba']        # 각 IC의 '최댓값' 확률 (1D, 길이 n_components)
labels = result['labels']             # 각 IC의 최빈 라벨 문자열

# 전체 7클래스 확률 행렬이 필요하면 iclabel 하위 함수로 직접:
from mne_icalabel.iclabel import iclabel_label_components
proba_full = iclabel_label_components(cleaned, ic)   # shape (n_components, 7)

eye = proba_full[:, CLASSES.index('eye')]            # Eye 확률만
print(f'\n{SUBJ.stem}  IC별 Eye 확률(내림차순):')
for i in np.argsort(eye)[::-1]:
    print(f'  IC{i:02d}  eye={eye[i]:.2f}   (최빈라벨={labels[i]})')