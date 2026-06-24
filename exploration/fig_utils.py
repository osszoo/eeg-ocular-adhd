"""전처리 단계별 figure를 results/에 저장하고 진행 로그(README)를 자동 갱신하는 헬퍼.

단계별 전처리는 각자 다른 채팅/스크립트에서 진행되므로, 어느 explore 작업에서든
동일하게 호출할 수 있도록 설계했다. 사용 예:

    from fig_utils import save_fig
    save_fig(fig, order=1, name="bandpass", note="bandpass 진행 후 PSD. 40Hz 위 신경 신호 없음 확인.")

동작:
  1. results/figures/01_bandpass.png 로 그림 저장 (같은 order면 덮어씀)
  2. results/README.md 에 해당 항목 추가/갱신 (order 순서 유지)

깃허브 업로드(commit/push)는 하지 않는다. 평소처럼 VS Code 소스 컨트롤에서 직접 올린다.
"""

from pathlib import Path
import re

# 이 파일은 exploration/ 안에 있고, 레포 루트는 그 부모.
# cwd와 무관하게 항상 같은 위치를 가리키도록 __file__ 기준으로 경로를 잡는다.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"
_README = _RESULTS_DIR / "README.md"

_HEADER = "# 전처리 진행 로그\n\n각 전처리 단계의 figure와 메모. `fig_utils.save_fig()`로 자동 갱신됨.\n"


def save_fig(fig, order, name, note, dpi=150):
    """figure를 저장하고 진행 로그를 갱신한다.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        저장할 그림 객체.
    order : int
        단계 번호. 파일명 prefix와 로그 정렬에 쓰인다 (예: 1 -> 01_).
    name : str
        단계 이름. 파일명과 로그 제목에 쓰인다 (예: "bandpass").
    note : str
        진행 메모. 로그 본문에 들어간다 (예: "bandpass 진행 후 ...").
    dpi : int
        저장 해상도.
    """
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fname = f"{order:02d}_{name}.png"
    fpath = _FIGURES_DIR / fname
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")

    _update_readme(order, name, note, fname)
    print(f"[save_fig] saved -> {fpath.relative_to(_REPO_ROOT)}")
    print(f"[save_fig] log updated -> {_README.relative_to(_REPO_ROOT)}")
    print("[save_fig] commit/push는 VS Code 소스 컨트롤에서 직접.")


def _build_block(order, name, note, fname):
    """단일 단계 항목 블록. 주석 마커로 감싸 나중에 찾아 갱신한다."""
    return (
        f"<!-- fig:{order:02d} -->\n"
        f"## {order:02d}. {name}\n\n"
        f"{note}\n\n"
        f"![{name}](figures/{fname})\n"
        f"<!-- /fig:{order:02d} -->"
    )


def _update_readme(order, name, note, fname):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    text = _README.read_text(encoding="utf-8") if _README.exists() else _HEADER

    block = _build_block(order, name, note, fname)
    marker = f"<!-- fig:{order:02d} -->"

    if marker in text:
        # 같은 order 블록 덮어쓰기
        pattern = re.compile(
            rf"<!-- fig:{order:02d} -->.*?<!-- /fig:{order:02d} -->",
            re.DOTALL,
        )
        text = pattern.sub(block, text)
    else:
        # 기존 블록들을 모두 모은 뒤 새 블록 포함해 order 순으로 재정렬
        blocks = re.findall(r"<!-- fig:\d+ -->.*?<!-- /fig:\d+ -->", text, re.DOTALL)
        blocks.append(block)

        def _order_of(b):
            return int(re.search(r"<!-- fig:(\d+) -->", b).group(1))

        blocks = sorted(set(blocks), key=_order_of)
        text = _HEADER + "\n" + "\n\n".join(blocks) + "\n"

    _README.write_text(text, encoding="utf-8")