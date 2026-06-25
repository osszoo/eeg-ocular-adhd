"""기록 전용 스크립트: figures.py 로 그림을 만들어 results/ 에 저장한다.

실행하면 results/figures/*.png 와 results/README.md(진행 로그)가 갱신된다.
화면에는 아무것도 띄우지 않는다(plt.show 없음) — 순수 기록용이다.

역할 분담:
  - figures.py     : 그림을 만든다 (fig 반환)
  - fig_utils.py   : 그림을 저장한다 (png 저장 + README 갱신) ← save_fig()
  - save_figures.py: 무엇을, 몇 번(order)으로, 어떤 메모와 함께 저장할지 정한다 (이 파일)

깃허브 반영은 자동이 아니다. 실행 후 결과를 확인하고, 평소처럼 VS Code
소스 컨트롤에서 results/ 변경분을 직접 commit·push 한다.

order 번호 체계 (파일명 prefix·README 정렬 기준):
  01 group_psd     [블록 A] 그룹 PSD
  02 psd_raw       [블록 B] ① 원본
  03 psd_bandpass  [블록 B] ② 밴드패스
  04 psd_car       [블록 B] ③ 밴드패스+CAR
  05 psd_stages    [블록 B] ④ 단계 오버레이
  06~ 이후 ASR·ICA 등 다음 단계에서 이어붙인다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from figures import make_group_psd, make_stage_psds
from fig_utils import save_fig

# ── [블록 A] 그룹 PSD ───────────────────────────────────────────────────────
save_fig(make_group_psd(), order=1, name='group_psd',
         note='ADHD 4 + Control 4 원본 PSD. 40Hz 위 공통 신경신호 없음 → H_FREQ=40 근거. 50Hz 라인노이즈 확인.')

# ── [블록 B] 단계별 PSD (동일 피험자 v10p) ──────────────────────────────────
# 4장이 같은 처리 결과에서 나오므로 make_stage_psds() 를 한 번만 호출해 dict로 받는다.
figs = make_stage_psds()
save_fig(figs['raw'],      order=2, name='psd_raw',
         note='① 원본 PSD(v10p). 40Hz 위까지 파워 이어지고 50Hz 라인노이즈 보임.')
save_fig(figs['bandpass'], order=3, name='psd_bandpass',
         note='② 밴드패스 후. 40Hz 위 절단, 0.5Hz 미만 표류 정리됨.')
save_fig(figs['car'],      order=4, name='psd_car',
         note='③ 밴드패스+CAR 후. 공통성분만큼 1–30Hz 하강, 알파 봉우리 유지.')
save_fig(figs['overlay'],  order=5, name='psd_stages',
         note='④ 원본·밴드패스·+CAR 오버레이. 고주파 절단과 전체 하강을 한눈에 비교.')

print('\n[save_figures] 완료. results/ 변경분을 VS Code 소스 컨트롤에서 commit·push 하세요.')