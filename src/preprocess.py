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
    import numpy as np

    # ADHD(part1)와 Control 폴더에서 각각 앞쪽 몇 명씩 뽑아 PSD 비교
    n_each = 4
    groups = {
        'ADHD':    sorted(Path('ADHD_part1').glob('*.mat'))[:n_each],
        'Control': sorted(Path('Control_part1').glob('*.mat'))[:n_each],
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {'ADHD': 'tab:red', 'Control': 'tab:blue'}

    for label, files in groups.items():
        for f in files:
            raw = to_raw(load_subject(f))
            psd = raw.compute_psd(fmax=64)
            freqs = psd.freqs
            # 채널 평균 후 dB 변환 — 한 사람당 곡선 하나
            mean_power = 10 * np.log10(psd.get_data().mean(axis=0))
            ax.plot(freqs, mean_power, color=colors[label],
                    alpha=0.6, linewidth=0.8,
                    label=label if f == files[0] else None)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'PSD: ADHD vs Control (각 {n_each}명, 필터 전)')
    ax.axvline(40, color='gray', linestyle='--', linewidth=0.8)  # 로우패스 후보선
    ax.axvline(50, color='black', linestyle=':', linewidth=0.8)  # 라인노이즈 위치
    ax.legend()
    plt.tight_layout()
    plt.show()