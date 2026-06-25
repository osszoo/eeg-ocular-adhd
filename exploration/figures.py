"""그림 생성 전용 모듈.

각 함수는 matplotlib Figure 객체를 '만들어서 반환'하기만 한다.
화면 표시(plt.show)도 파일 저장(save_fig)도 여기서 하지 않는다 —
그 둘은 이 모듈을 호출하는 쪽이 결정한다.

  - exploration/explore.py      : 화면에 띄워 탐색·확인만 (results/ 저장 안 함)
  - exploration/save_figures.py : results/ 에 저장해 기록만 (화면 안 띄움)

'그림 생성'을 이 한 곳에만 두는 이유: 탐색용과 기록용이 같은 그림을
필요로 하는데, 각 파일에서 따로 그리면 코드가 복제되고 한쪽만 고쳤을 때
어긋난다. 생성 함수를 단일 출처로 두고 두 진입점이 import해 쓴다.

아래 두 그림 함수는 '서로 다른 질문'에 답하는 별개의 그림이다.
  make_group_psd()  : "로우패스 상한을 40Hz로 잡아도 되나?"   → 여러 명, 원본만
  make_stage_psds() : "전처리 단계가 신호를 의도대로 바꿨나?"  → 한 명, 단계별
둘 다 '원본 PSD'를 포함하지만 대상이 달라(8명 vs 1명) 중복이 아니며,
한쪽이 다른 쪽을 대체하지 못한다.
"""
import sys
from pathlib import Path

# src/ 를 import 경로에 추가 (exploration/ 과 src/ 가 프로젝트 루트에 나란히 있는 구조).
# __file__ 기준이라 어떤 cwd에서 실행하든 경로가 안 깨진다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_subject, to_raw   # 로딩은 load_data 단일 출처
from preprocess import bandpass, car          # 확정된 파이프라인 함수 재사용

# 한글 폰트 설정은 그림을 만드는 이 모듈에서 한 번만 둔다.
# explore.py / save_figures.py 는 이 모듈을 import하므로 별도 설정이 필요 없다.
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 (경고 제거)
plt.rcParams['axes.unicode_minus'] = False       # 음수 축 라벨 깨짐 방지


def _db(psd):
    """PSD 객체 → 채널평균 파워를 dB로 변환한 1차원 배열.
    dB = 10*log10(power). 작은 파워값을 로그로 압축해 1/f 모양과 봉우리를
    한 축에서 같이 보기 위함(그래서 값이 음수로 나오는 게 정상)."""
    return 10 * np.log10(psd.get_data().mean(axis=0))


def make_group_psd(n_each=4):
    """[블록 A] 그룹 PSD 한 장 반환 — 로우패스 상한(H_FREQ=40) 근거 확인용.

    목적 : 로우패스 상한 결정 근거. 개체차를 넘어선 '공통 패턴'을 봐야 하므로
           여러 명(ADHD n_each + Control n_each)을 겹쳐 그린다. 신호는 전부 필터 전 원본.
    판독 : 여러 명에서 공통으로 40Hz 위에 의미있는 신경신호가 없고(→ 상한 40 타당),
           50Hz에 라인노이즈 봉우리가 보이면 H_FREQ=40 확정 근거가 된다.
    주의 : 여기 대상은 (기본) 8명. 단계 효과가 아니라 '주파수 상한'을 보는 그림이다.
    """
    groups = {
        'ADHD':    sorted(Path('ADHD_part1').glob('*.mat'))[:n_each],
        'Control': sorted(Path('Control_part1').glob('*.mat'))[:n_each],
    }
    colors = {'ADHD': 'tab:red', 'Control': 'tab:blue'}

    fig, ax = plt.subplots(figsize=(11, 4))
    for label, files in groups.items():
        for f in files:
            raw = to_raw(load_subject(f))
            psd = raw.compute_psd(fmax=64)
            # 같은 라벨은 색을 공유 → 범례 중복을 막으려 그룹당 첫 파일에만 label을 단다.
            ax.plot(psd.freqs, _db(psd), color=colors[label],
                    alpha=0.6, linewidth=0.8,
                    label=label if f == files[0] else None)
    ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
           title=f'PSD: ADHD vs Control (각 {n_each}명, 필터 전)')
    ax.axvline(40, color='gray',  linestyle='--', linewidth=0.8)   # 로우패스 상한 후보
    ax.axvline(50, color='black', linestyle=':',  linewidth=0.8)   # 전원 라인노이즈
    ax.legend()
    fig.tight_layout()
    return fig


def make_stage_psds():
    """[블록 B] 단계별 PSD 4장 반환 — 전처리 단계 검증용. dict로 반환.

    목적 : 각 전처리 단계(원본 → bandpass → +CAR)가 신호를 의도대로 바꿨는지 확인.
           단계 '효과'를 보려면 피험자를 고정해야 한다(사람이 바뀌면 변화가 전처리
           때문인지 개체차 때문인지 구분 불가). 그래서 동일 1명(v10p)만 사용한다.
    판독 : ① 원본은 40Hz 위까지 파워가 이어지고 50Hz 라인노이즈가 보인다.
           ② bandpass 후 40Hz 위가 깎이고 0.5Hz 미만 표류가 정리된다.
           ③ +CAR 후 모양은 ②와 거의 같되 공통성분만큼 1–30Hz가 몇 dB 내려간다.
           ④ 셋을 겹쳐 '고주파 절단(①→②)'과 '전체 하강(②→③)'을 한눈에 본다.
    주의 : 블록 A와 달리 여기 대상은 1명. 일반화 근거가 아니라 단계 검증용이다.

    반환 : {'raw', 'bandpass', 'car', 'overlay'} 키를 가진 dict (각 값이 Figure).

    구현 메모 : ①②③④는 모두 같은 피험자의 같은 처리 결과에서 나온다. 그래서
    로드·전처리·PSD 계산을 함수 안에서 '한 번만' 하고 그 결과로 4장을 그린다.
    (각 그림마다 다시 처리하면 중복 연산이고, 미묘한 불일치 위험도 생긴다.)
    """
    f = sorted(Path('ADHD_part1').glob('*.mat'))[0]   # 단계 비교용 동일 피험자(v10p)
    raw   = to_raw(load_subject(f))   # 원본 (필터 전)
    filt  = bandpass(raw)             # 1단계: 밴드패스
    reref = car(filt)                 # 2단계: CAR (밴드패스 출력에 적용)

    # 세 단계 PSD를 미리 계산 (모두 동일 피험자에서 파생 → 직접 비교 가능)
    psd_raw  = raw.compute_psd(fmax=64)
    psd_filt = filt.compute_psd(fmax=64)
    psd_car  = reref.compute_psd(fmax=64)

    def _single(psd, title, color):
        """단계 하나의 PSD를 한 장으로 그려 반환하는 내부 헬퍼."""
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(psd.freqs, _db(psd), color=color, linewidth=1.3)
        ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)', title=title)
        fig.tight_layout()
        return fig

    fig_raw  = _single(psd_raw,  '① 원본 PSD (필터 전, v10p)',     'tab:blue')
    fig_filt = _single(psd_filt, '② 밴드패스 후 PSD (v10p)',       'tab:gray')
    fig_car  = _single(psd_car,  '③ 밴드패스 + CAR 후 PSD (v10p)', 'tab:green')

    # ④ 세 단계를 한 축에 겹쳐 '단계 간 변화'를 직접 비교
    fig_all, ax = plt.subplots(figsize=(11, 4))
    ax.plot(psd_raw.freqs,  _db(psd_raw),  color='tab:blue',  linewidth=1.3, label='① 원본')
    ax.plot(psd_filt.freqs, _db(psd_filt), color='tab:gray',  linewidth=1.3, label='② 밴드패스')
    ax.plot(psd_car.freqs,  _db(psd_car),  color='tab:green', linewidth=1.3, label='③ 밴드패스+CAR')
    ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
           title='④ 단계별 PSD 비교 (동일 피험자 v10p)')
    ax.legend()
    fig_all.tight_layout()

    return {'raw': fig_raw, 'bandpass': fig_filt, 'car': fig_car, 'overlay': fig_all}