"""Render the final comparison only after raw 9B evaluation verification exists."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/content/local_fast/lsdrag-text-v1')


def read(path):
    return json.loads(path.read_text())


def main():
    old_path = REPO / 'lab/reports/qwen35_9b_historical_reference_20261024/verification.json'
    new_path = REPO / 'lab/reports/qwen35_9b_h3_validation64_20261024/verification.json'
    old, new = read(old_path), read(new_path)
    run = ROOT / 'results/qwen35_9b_h3_validation64_20261024'
    status, spec = read(run / 'status.json'), read(run / 'spec.json')
    assert status['phase'] == 'complete' and status['episodes'] == 192
    assert old['verified'] and new['verified']
    assert old['qids'] == spec['qids'] and len(spec['qids']) == 64
    # Reports are evidence only while the raw episodes they certify still match.
    for report in (old, new):
        for path, digest in report['episode_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for policy in spec['policies'].values():
        assert hashlib.sha256(Path(policy['checkpoint']).read_bytes()).hexdigest() == policy['sha256']
    f = lambda value: '—' if value is None else f'{value:.2f}'
    lines = [
        '# Qwen3.5-9B H3 일반화 실험 결과', '',
        '같은 개발64문항에서 각 모델의 base→H3 변화를 비교한다. '
        '9B는 새 teacher와 행동별 후속 결과로 base를 학습한 뒤 H3와 상수 종료 대조군을 학습했다. '
        '아래 표는 teacher 학습점수가 아닌 실제 정책 실행 결과다.', '',
        '| 모델 | 정책 | F1(%) | EM 정답/64 | 평균 생성토큰 | 평균 도구호출 | 강제종료 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    rows = [('4B', name, p) for name, p in old['policies'].items()]
    rows += [('9B', name, p) for name, p in new['policies'].items()]
    for model, name, p in rows:
        lines.append(f"| {model} | {name} | {100*p['f1']:.2f} | {p['correct']['n']} | {f(p['all']['mean_generated'])} | {p['mean_tools']:.2f} | {p['forced_final']} |")
    lines += ['', '| 비교 | F1 변화(pp) | 생성토큰 변화(%) |', '|---|---:|---:|']
    effects = [('4B H3 − base', old)] + [('9B ' + k, v) for k, v in new['paired'].items()]
    for name, p in effects:
        lines.append(f"| {name} | {100*p['f1_delta']:+.2f} | {100*(p['token_ratio']-1):+.2f} |")
    lines += ['', '## 정답·오답별 생성 비용', '',
              '공식 EM으로 분류한다. 부분 F1 오답은 오답 전체에 포함되며 별도 행은 그 부분집합이다.', '',
              '| 모델 | 정책 | 분류 | 문항 | 총 생성토큰 | 평균 생성토큰 |',
              '|---|---|---|---:|---:|---:|']
    for model, name, p in rows:
        for key, label in [('correct', '정답'), ('incorrect', '오답'), ('partial', '부분F1 오답')]:
            s = p[key]
            lines.append(f"| {model} | {name} | {label} | {s['n']} | {s['total_generated']} | {f(s['mean_generated'])} |")
    lines += ['', '## 9B 문항별 정오 변화와 비용', '',
              '절약 토큰은 기준 정책 총량−후보 정책 총량이다. 양수는 절약이다.', '',
              '| 후보 vs 기준 | 분류 | 문항 | 기준 토큰 | 후보 토큰 | 절약 토큰 |',
              '|---|---|---:|---:|---:|---:|']
    for name, comparison in new['paired'].items():
        for group, v in comparison['groups'].items():
            lines.append(f"| {name} | {group} | {v['reference']['n']} | {v['reference']['total_generated']} | {v['candidate']['total_generated']} | {v['saved_tokens']} |")
    lines += ['', '## 생성량과 해석 범위', '',
              '| 모델 | 정책 | 생성 문자 총량 | 생성 UTF-8 바이트 총량 |',
              '|---|---|---:|---:|']
    for model, name, p in rows:
        lines.append(f"| {model} | {name} | {p['output_characters']} | {p['output_utf8_bytes']} |")
    lines += ['',
              f"9B 실행 오류가 있는 궤적: {new['errors']}. 오류가 있으면 정책 품질과 실행 실패를 구분해 해석해야 한다.", '',
              'H3는 동결된 reader/base 위에서 미래 답변 품질과 생성비용으로 학습한 종료 보정이다. '
              'critic이나 전체 행동 계획 능력을 입증하는 실험은 아니다. '
              '상수 종료 대조군보다 같은 비용에서 더 높은 정확도를 보이는지 검토해야 하며, 상수 대조군의 비용이 일치한다고 가정하지 않는다.', '',
              '4B base의 과거 탐색적 학습 이력과 새9B base 준비는 동일하지 않다. '
              '모델 세대·크기·구조·토크나이저도 함께 달라져 순수 크기 효과를 주장할 수 없다. '
              '토큰 수는 모델별 단위이므로 모델 내부 비율을 우선 해석한다. '
              '공통 JSON 프로토콜의9B이며 수정 없는 native XML 기준이 아니다. '
              '반복 노출된 개발64 결과이며 독립 검증이나 통제된 지연시간 결과는 아니다. reserved80과 기존 final80은 사용하지 않았다.', '',
              '## 검증 출처', '',
              f'- `{old_path.relative_to(REPO)}`',
              f'- `{new_path.relative_to(REPO)}`',
              '- `docs/JEV_QWEN35_9B_H3_GENERALIZATION_20261024.md`', '',
              '자동 집계표다. 목표 완료 판정 전 학습 단계 검증·백업·원본 및 해석 검토를 별도로 수행한다.',
    ]
    dest = REPO / 'docs/JEV_QWEN35_9B_H3_RESULTS_20261024.md'
    dest.write_text('\n'.join(lines) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
