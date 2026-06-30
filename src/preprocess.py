"""전처리 파이프라인.

적재된 Raw를 다음 순서로 처리한다:
  1. 밴드패스 필터 (bandpass)  : 0.5–40Hz 대역으로 제한
  2. 공통평균참조  (car)       : 평균참조로 재참조, rank 19→18 → ICA n_components=18
  3. ASR           (asr)       : 큰 진폭 버스트 아티팩트를 부분공간 재구성으로 정리
  4. ICA           (ica)       : extended Infomax로 IC 분해(피험자별 rank만큼)

이후 단계 ICLabel는 이 파일에 추가 예정.
"""
from pathlib import Path
import mne
from load_data import load_subject, to_raw

L_FREQ = 0.5      # 하이패스 하한: 느린 표류 제거, 안구 저주파 보존
H_FREQ = 40.0     # 로우패스 상한: PSD 확인으로 확정 (50Hz 라인노이즈 차단 겸)

# ASR cutoff: 안구 성분 보존이 1순위라 보수적으로 둔다.
# asrpy는 EEGLAB 모던 스케일 — 클수록 보수적(덜 제거), 작을수록 공격적, 기본 20.
# meegkit 레거시 스케일(기본 5, 공격적 2.5)과 혼동 금지: 작은 값을 넣으면 안구까지 깎인다.
# 확정값(100): asr_sweep(재구성률·전두 분산)으로 20·30 탈락 → off/100/50을
# cutoff_compare(ICA→ICLabel)로 비교, 셋 다 rank 18·안구 IC 보존 확인. 짧은 녹화에서
# 50은 재구성률이 튀어 가장 보수적인 100 선택(안구 보존 우선).
ASR_CUTOFF = 100


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
    ICLabel이 평균참조 데이터로 학습됐으므로 ICA 입력을 여기에 맞춘다.
    """
    return raw.copy().set_eeg_reference('average', projection=False)


# ── 전처리 3단계: ASR (Artifact Subspace Reconstruction) ──────
def asr(raw, cutoff=ASR_CUTOFF):
    """큰 진폭 버스트 아티팩트를 부분공간 재구성으로 정리한다.

    asrpy 자기보정(self-calibration): 기록에서 깨끗한 구간을 로버스트 통계로
    자동 추정해 기준 공분산을 잡고, 그보다 분산이 cutoff(SD)배 이상 튀는 구간을
    아티팩트로 보고 성한 채널들로부터 그 부분공간을 재구성한다(구간을 잘라내지
    않으므로 타임라인·녹화 길이가 보존됨 → rate 기반 안구 특징 설계와 호환).
    cutoff가 클수록 보수적(덜 제거) — 안구 성분 보존을 위해 크게 둔다.

    이 단계는 CAR 뒤라 입력이 rank 18(채널합=0 제약)이다. asrpy는 0에 가까운
    공간방향을 자동 배제하도록 설계돼 rank-deficient 입력에서도 동작한다.
    (full-rank가 추정엔 더 이상적이라는 점은 스윕 때 순서 비교로 확인 예정.)
    원본 보존 위해 copy() 후 적용.
    """
    import asrpy   # 무거운 의존성이라 함수 호출 시에만 import
    raw = raw.copy()
    model = asrpy.ASR(sfreq=raw.info['sfreq'], cutoff=cutoff)
    model.fit(raw)               # 깨끗한 구간으로 기준 공분산 학습
    return model.transform(raw)  # 버스트 구간 재구성


# ── 전처리 4단계: ICA(독립성분분석) ──────────────────────────
def ica(raw, random_state=97):
    """ASR까지 끝난 Raw를 extended Infomax로 IC 분해한다(피험자 단위).

    안구 성분을 신경 성분과 분리하기 위한 핵심 단계. 여기서는 '분해'만 하고
    라벨링(ICLabel)은 별도 단계에서 그 결과 위에 얹는다.

    - 알고리즘 : extended Infomax 고정. ICLabel이 이 방식으로 분해된 IC로
                 학습됐기 때문에 호환을 위해 다른 알고리즘(FastICA·Picard)을 쓰지 않는다.
    - n_components : 하드코딩(18)하지 않고 ASR 후 데이터의 rank로 피험자마다 정한다.
                     CAR가 rank를 19→18로 떨어뜨리고(채널합=0 제약), ASR이 더 깎을 수
                     있어 경험적으로 재확인한다(compute_rank는 ASR이 만드는 '거의 0'인
                     미세 고유값을 허용오차로 처리). rank보다 많은 성분을 요구하면
                     유령 성분(한 소스가 둘로 쪼개짐)이 생겨 라벨링을 흔든다.
    - 적합 대상 : 0.5Hz cleaned 데이터 그대로(1Hz 사본 미사용). 표류는 이미
                  0.5Hz 하이패스+ASR로 제거됐고, 안구 저주파를 ICA가 온전히 보게 한다.
    - random_state : Infomax는 확률적이라 재현성을 위해 시드를 고정한다.

    fit 끝난 ICA 객체를 반환한다(원본 raw는 건드리지 않음).
    """
    raw = raw.copy()                              # 원본 보존
    rank = mne.compute_rank(raw)['eeg']           # ASR 후 실제 rank(허용오차 기반)

    fitted = mne.preprocessing.ICA(
        n_components=rank,                        # 피험자별 rank만큼만 잡음
        method='infomax',
        fit_params=dict(extended=True),           # ICLabel 호환: extended Infomax
        max_iter='auto',
        random_state=random_state,
    )
    fitted.fit(raw)
    return fitted


if __name__ == '__main__':
    arr = load_subject(sorted(Path('ADHD_part1').glob('*.mat'))[0])
    raw = to_raw(arr)
    filt = bandpass(raw)
    reref = car(filt)
    cleaned = asr(reref)
    print('밴드패스 → CAR → ASR 통과:',
          round(cleaned.n_times / cleaned.info['sfreq'], 1), '초')
    ic = ica(cleaned)
    print('ICA 통과: n_components(=ASR 후 rank) =', ic.n_components_)