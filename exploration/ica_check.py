"""ICA 검증(진단 전용, 단일 피험자): IC 분해가 안구 성분을 제대로 잡는지 눈으로 확인.

무겁고(extended Infomax) preprocess를 직접 import하므로, explore.py의 '적재 점검 +
PSD' 성격과 분리한다(asr_sweep.py와 동일한 이유).

확인할 세 가지.
  1. n_components(=ASR 후 rank) : CAR로 18이 기본. ASR이 더 깎아 17·16이면 차원을
                                  통째로 없앤다는 신호 → cutoff 비교의 한 지표로 메모.
  2. plot_components 토포맵       : 안구 IC는 앞쪽(전두) 집중으로 보여야 한다.
  3. plot_sources 시간경로        : 안구 IC는 느린 깜빡임 버스트가 보여야 한다.

잠근 순서(bandpass → CAR → ASR → ICA) 그대로 단일 피험자(v10p)에 적용한다.
ASR cutoff는 asr() 기본값(20). cutoff 후보(off/100/50) 비교는 이게 한 명에서
도는 걸 확인한 뒤 별도로 진행한다. 실행하면 토포맵·시간경로 창이 뜬다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import matplotlib.pyplot as plt
from load_data import load_subject, to_raw
from preprocess import bandpass, car, asr, ica   # 잠근 파이프라인 함수 재사용(단일 출처)

# 이전 단계들과 동일 피험자(v10p)로 — 비교 가능성 유지
SUBJ = next(p for d in ['ADHD_part1', 'ADHD_part2', 'Control_part1', 'Control_part2']
            for p in Path(d).glob('v10p.mat'))

cleaned = asr(car(bandpass(to_raw(load_subject(SUBJ)))))
ic = ica(cleaned)

print(f'{SUBJ.stem}  n_components(=ASR 후 rank) = {ic.n_components_}')
ic.plot_components()        # IC별 토포맵 — 안구 IC는 앞쪽 집중
ic.plot_sources(cleaned)    # IC별 시간경로 — 안구 IC는 깜빡임 버스트
plt.show()