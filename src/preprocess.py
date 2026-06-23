"""전처리 파이프라인.

적재된 Raw를 다음 순서로 처리한다:
  1. 밴드패스 필터 (bandpass)  : 0.5–40Hz 대역으로 제한
  2. 공통평균참조  (car)       : 평균참조로 재참조, rank 19→18

이후 단계(ASR → ICA → ICLabel)는 차례로 이 파일에 추가 예정.
(ICA용 1Hz 사본은 이 단계가 아니라 이후 ICA 단계에서 따로 생성)
"""
from pathlib import Path
from load_data import load_subject, to_raw

L_FREQ = 0.5      # 하이패스 하한: 느린 표류 제거, 안구 저주파 보존
H_FREQ = 40.0     # 로우패스 상한: PSD 확인으로 확정 (50Hz 라인노이즈 차단 겸)


# ── 전처리 1단계: 밴드패스 필터 ───────────────────────────────
def bandpass(raw, l_freq=L_FREQ, h_freq=H_FREQ):
    """적재된 Raw를 최종 분석 대역(0.5–40Hz)으로 필터링한다.
    zero-phase FIR. 원본 보존 위해 copy() 후 필터.
    """
    return raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method='fir', phase='zero', fir_design='firwin')


# ── 전처리 2단계: 공통평균참조(CAR) ──────────────────────────
def car(raw):
    """매 시점 19채널 평균을 빼 평균참조로 재참조한다.
    rank를 1 줄임(19→18) → ICA n_components=18. 원본 보존 위해 copy() 후 재참조.
    """
    return raw.copy().set_eeg_reference('average', projection=False)


if __name__ == '__main__':
    arr = load_subject(sorted(Path('ADHD_part1').glob('*.mat'))[0])
    raw = to_raw(arr)
    filt = bandpass(raw)
    reref = car(filt)
    print('밴드패스 → CAR 통과:', round(reref.n_times / reref.info['sfreq'], 1), '초')