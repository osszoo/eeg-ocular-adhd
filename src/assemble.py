"""조건별 특징 표 조립(정식 파이프라인).

features.py가 만든 두 원천 표를 네 조건의 분류용 표로 조립한다.

입력(results/):
  - features_ocular.csv : 84명, [subject_id, label, ocular_power_ratio]
  - features_neural.csv : 121명, [subject_id, label, <뇌파 25열>]

출력(results/):
  - X_neural84.csv   : 84명, 뇌파 25            (주 neural-only)
  - X_ocular84.csv   : 84명, 안구 1             (주 ocular-only)
  - X_combined84.csv : 84명, 뇌파 25 + 안구 1   (주 combined, 핵심 비교)
  - X_neural121.csv  : 121명, 뇌파 25           (보조 baseline)

84명 정의 : features_ocular.csv의 subject_id 집합(안구 IC 보유자).
정렬 검증 : combined는 뇌파·안구 행을 '순서 신뢰'로 결합하므로, 그 전제를
       피험자별 행수 일치 + 라벨 일치로 엄격히 검증한다. 불일치 시 조립 중단.
       (두 표 모두 같은 신호를 같은 make_epochs로 잘라 순서·개수가 같아야 정상.)

한 행 = 6초 에폭. z-정규화는 여기서 하지 않는다(train fold 전용, 분류 단계로).
"""
import sys
import csv
from pathlib import Path
from collections import OrderedDict

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / 'results'
OCULAR_CSV = _RESULTS / 'features_ocular.csv'
NEURAL_CSV = _RESULTS / 'features_neural.csv'
OUT_PATHS = {
    'neural84':   _RESULTS / 'X_neural84.csv',
    'ocular84':   _RESULTS / 'X_ocular84.csv',
    'combined84': _RESULTS / 'X_combined84.csv',
    'neural121':  _RESULTS / 'X_neural121.csv',
}


# ── csv 로드/그룹 ────────────────────────────────────────────
def load_csv(path):
    """csv → (header, rows). rows는 문자열 리스트."""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def group_by_subject(rows):
    """등장 순서 유지하며 subject_id(0열)로 묶는다."""
    g = OrderedDict()
    for row in rows:
        g.setdefault(row[0], []).append(row)
    return g


# ── 조립(순수 함수: 파일 없이 테스트 가능) ──────────────────
def assemble(oc_header, oc_rows, ne_header, ne_rows):
    """두 원천을 네 조건 표로 조립. 반환 (outputs, report).

    outputs: {name: (header, rows)}
    report : {'errors': [...], 'subj84': N, 'subj121': N}
    errors가 비어야 정상. combined는 errors 있으면 만들지 않는다.
    """
    oc_by = group_by_subject(oc_rows)     # 84명
    ne_by = group_by_subject(ne_rows)     # 121명
    subj84 = list(oc_by.keys())           # 안구 csv 순서(= all_subjects 순서)
    errors = []

    # 검증 1: 84명이 모두 뇌파에 있나
    missing = [s for s in subj84 if s not in ne_by]
    if missing:
        errors.append(f'뇌파에 없는 안구 피험자: {missing}')

    # 검증 2: 피험자별 행수 일치 + 라벨 일치(순서 신뢰의 전제)
    for s in subj84:
        if s not in ne_by:
            continue
        n_oc, n_ne = len(oc_by[s]), len(ne_by[s])
        if n_oc != n_ne:
            errors.append(f'{s}: 에폭수 불일치 안구{n_oc} vs 뇌파{n_ne}')
        if oc_by[s][0][1] != ne_by[s][0][1]:
            errors.append(f'{s}: 라벨 불일치 안구{oc_by[s][0][1]} vs 뇌파{ne_by[s][0][1]}')

    outputs = {}
    # neural121 = 뇌파 그대로
    outputs['neural121'] = (ne_header, ne_rows)
    # ocular84 = 안구 그대로
    outputs['ocular84'] = (oc_header, oc_rows)
    # neural84 = 뇌파에서 84명만(안구 순서로)
    ne84_rows = [row for s in subj84 if s in ne_by for row in ne_by[s]]
    outputs['neural84'] = (ne_header, ne84_rows)

    # combined84 = 84명, 뇌파 25 + 안구 1 (순서 신뢰 결합). errors 없을 때만.
    if not errors:
        oc_feat_col = oc_header[2]                       # 'ocular_power_ratio'
        comb_header = ne_header + [oc_feat_col]
        comb_rows = []
        for s in subj84:
            for ne_row, oc_row in zip(ne_by[s], oc_by[s]):
                comb_rows.append(ne_row + [oc_row[2]])   # 뇌파행 + 안구값
        outputs['combined84'] = (comb_header, comb_rows)

    report = {'errors': errors, 'subj84': len(subj84), 'subj121': len(ne_by)}
    return outputs, report


# ── 저장/요약 ────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def summarize(name, header, rows):
    subjects = {r[0] for r in rows}
    nan = 0
    for r in rows:
        for v in r[2:]:                    # 특징 열만(subject_id·label 제외)
            try:
                x = float(v)
                if x != x:
                    nan += 1
            except ValueError:
                nan += 1
    print(f'  {name:<12} {len(rows):>5}행 × {len(header):>2}열 · '
          f'{len(subjects):>3}명 · NaN {nan}')


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    oc_header, oc_rows = load_csv(OCULAR_CSV)
    ne_header, ne_rows = load_csv(NEURAL_CSV)

    outputs, report = assemble(oc_header, oc_rows, ne_header, ne_rows)

    print('=== 조건별 표 조립 ===')
    print(f'안구 84명(기대 84): {report["subj84"]} · 뇌파 121명(기대 121): {report["subj121"]}')

    if report['errors']:
        print('\n⚠ 정렬 검증 실패 — combined 미생성. 원인:')
        for e in report['errors'][:20]:
            print(f'  - {e}')
        print('\n뇌파·안구 조건만 저장하고 combined는 건너뜁니다.')
    else:
        print('정렬 검증 통과: 84명 전원 안구·뇌파 에폭수·라벨 일치.\n')

    # 저장 + 요약
    print('[저장·요약]')
    for name, path in OUT_PATHS.items():
        if name not in outputs:
            continue
        header, rows = outputs[name]
        write_csv(path, header, rows)
        summarize(name, header, rows)

    # 일관성 교차 점검
    if 'combined84' in outputs:
        n_oc = len(outputs['ocular84'][1])
        n_ne84 = len(outputs['neural84'][1])
        n_comb = len(outputs['combined84'][1])
        n_ne121 = len(outputs['neural121'][1])
        print('\n[교차 점검]')
        print(f'  안구84 = 뇌파84 = combined84 (모두 같아야): '
              f'{n_oc} / {n_ne84} / {n_comb} '
              f'{"✓" if n_oc == n_ne84 == n_comb else "✗ 불일치"}')
        print(f'  뇌파121 총행: {n_ne121} (= 뇌파84 {n_ne84} + 나머지 {n_ne121 - n_ne84})')
        comb_cols = len(outputs['combined84'][0])
        print(f'  combined 열수: {comb_cols} (기대 28 = subject_id+label+뇌파25+안구1)')