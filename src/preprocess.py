"""전처리 1단계: 밴드패스 필터.
적재된 Raw를 최종 분석 대역으로 필터링한다.
(ICA용 1Hz 사본은 이 단계가 아니라 이후 ICA 단계에서 따로 생성)"""
import matplotlib.pyplot as plt
from pathlib import Path
from load_data import load_subject, to_raw

L_FREQ = 0.5     # 하이패스 하한: 느린 표류 제거, 안구 저주파는 보존
H_FREQ = 40.0    # 로우패스 상한: 잠정값 — 아래 PSD 확인 후 확정

def bandpass(raw, l_freq=L_FREQ, h_freq=H_FREQ):
    """zero-phase FIR 밴드패스. 원본 보존 위해 copy() 후 필터."""
    return raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method='fir', phase='zero', fir_design='firwin')

if __name__ == '__main__':
    arr = load_subject(sorted(Path('ADHD_part1').glob('*.mat'))[0])
    raw = to_raw(arr)

    # 필터 전 PSD: 50Hz 라인노이즈 유무 + 고주파 감쇠 지점 확인
    raw.compute_psd(fmax=64).plot()
    plt.show()

    filt = bandpass(raw)
    print('=== 밴드패스 검증 ===')
    print(f'대역: {L_FREQ}–{H_FREQ} Hz, FIR zero-phase')
    print('필터 후 길이(초):', round(filt.n_times / filt.info['sfreq'], 1))