"""적재(loading): .mat → 위치정보가 붙은 MNE Raw 객체.
신호 값은 가공하지 않고 메타정보(채널명·위치)만 부여한다."""
import scipy.io as sio
import numpy as np
import mne
from pathlib import Path

mne.set_log_level('WARNING')

SFREQ = 128
CH = ['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3',
      'F4','F7','F8','P3','P4','T5','T6','O1','O2']
RENAME = {'T3':'T7', 'T4':'T8', 'T5':'P7', 'T6':'P8'}   # 구형→현대 명명

def load_subject(mat_path):
    """.mat → (samples, 19) numpy 배열"""
    mat = sio.loadmat(mat_path)
    var = [k for k in mat if not k.startswith('__')][0]
    arr = np.asarray(mat[var], dtype=float)
    if arr.shape[0] == 19 and arr.shape[1] != 19:
        arr = arr.T
    return arr

def to_raw(arr):
    """(samples, 19) 배열 → 몽타주가 부착된 Raw 객체 (신호 미가공)"""
    info = mne.create_info(ch_names=CH, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(arr.T * 1e-6, info)    # (채널×샘플), µV→V 규약
    raw.rename_channels(RENAME)
    raw.set_montage('standard_1020')
    return raw

if __name__ == '__main__':
    FOLDERS = [('ADHD_part1', 1), ('ADHD_part2', 1),
               ('Control_part1', 0), ('Control_part2', 0)]

    ok, failed = [], []
    for folder, label in FOLDERS:
        for f in sorted(Path(folder).glob('*.mat')):
            try:
                arr = load_subject(f)
                raw = to_raw(arr)
                # 적재 무결성 점검
                pos = raw.get_montage().get_positions()['ch_pos']
                missing = [ch for ch in raw.ch_names
                           if np.isnan(list(pos[ch])).any()]
                assert len(raw.ch_names) == 19, '채널 수 ≠ 19'
                assert not missing, f'좌표 없는 채널 {missing}'
                ok.append((f.stem, label, raw.n_times / raw.info['sfreq']))
            except Exception as e:
                failed.append((f.stem, str(e)))

    print(f'=== 121명 일괄 적재 점검 ===')
    print(f'성공: {len(ok)}명 / 실패: {len(failed)}명\n')

    if failed:
        print('실패한 피험자:')
        for name, err in failed:
            print(f'  {name}: {err}')
    else:
        print('전원 적재 성공 (채널 19 + 좌표 매칭 모두 통과)')

    n_adhd = sum(lab == 1 for _, lab, _ in ok)
    n_ctrl = sum(lab == 0 for _, lab, _ in ok)
    print(f'\n적재 성공 내역: ADHD {n_adhd}명 / Control {n_ctrl}명')

    # 90초 미만 13명이 적재에도 문제없는지 따로 확인
    short = [(name, lab, round(dur, 1)) for name, lab, dur in ok if dur < 90]
    print(f'\n90초 미만 피험자 ({len(short)}명) — 적재는 정상:')
    for name, lab, dur in short:
        grp = 'ADHD' if lab == 1 else 'Control'
        print(f'  {name} ({grp}): {dur}s')