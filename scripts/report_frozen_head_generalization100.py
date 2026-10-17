"""CPU-only paired analysis of the frozen100 evaluation, no policy selection."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/frozen_head_generalization100_20261017'
OUT = REPO/'lab/reports/text_integrated_20261016'


def divergence(native, head):
    a, b = native['trajectory'], head['trajectory']
    for index, (n, h) in enumerate(zip(a, b)):
        if n['tool_name'] != h['tool_name']:
            return {'step': index, 'kind': 'action', 'native': n['tool_name'], 'head': h['tool_name']}
        if n['arguments'] != h['arguments']:
            return {'step': index, 'kind': 'arguments', 'action': n['tool_name'],
                    'native': n['arguments'], 'head': h['arguments']}
        if n['result'] != h['result']:
            return {'step': index, 'kind': 'tool_result'}
    if len(a) != len(b):
        return {'step': min(len(a), len(b)), 'kind': 'head_ends_earlier' if len(b) < len(a) else 'head_continues_longer',
                'warning': 'Endpoint difference can reflect termination or format error, not automatically premature stop.'}
    return {'kind': 'same_tool_trajectory', 'answer_changed': native['answer'] != head['answer']}


def stats(es):
    keys = ['f1', 'em', 'support_f1', 'tool_calls', 'interaction_turns', 'prompt_tokens', 'completion_tokens',
            'forced_prefix_tokens', 'prefill_count', 'decoder_forward_count_derived', 'seconds']
    times = np.array([e['seconds'] for e in es])
    return {'n': len(es), **{k: float(np.mean([e[k] for e in es])) for k in keys},
            'latency_total_seconds': float(times.sum()), 'latency_median_seconds': float(np.median(times)),
            'latency_p90_seconds': float(np.quantile(times, .9)),
            'error_episodes': sum(bool(e['errors']) for e in es),
            'valid_finals': sum(e['answer_format_valid'] for e in es),
            'forced_finals': sum(e['forced_final'] for e in es)}


def main():
    manifest = json.loads((RUN/'manifest.json').read_text())
    status = json.loads((RUN/'status.json').read_text())
    groups, hashes, records, pairs = {}, {}, [], []
    loaded = {}
    for policy in manifest['policies']:
        es = []
        for qid in manifest['qids']:
            path = RUN/'episodes'/policy/f'{qid}.json'
            if not path.exists():
                continue
            e = json.loads(path.read_text())
            assert e['tool_calls'] == len(e['trajectory']) and e['reused_argument_calls'] == 0
            hashes[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded[qid, policy] = e; es.append(e)
            records.append({k: e[k] for k in ['qid', 'policy', 'f1', 'em', 'support_f1', 'tool_calls', 'interaction_turns',
                                            'completion_tokens', 'forced_prefix_tokens', 'prompt_tokens', 'prefill_count',
                                            'decoder_forward_count_derived', 'seconds']})
        if es:
            groups[policy] = stats(es)
    for qid in manifest['qids']:
        if (qid, 'native') not in loaded or (qid, 'head_generate') not in loaded:
            continue
        n, h = loaded[qid, 'native'], loaded[qid, 'head_generate']
        pairs.append({'qid': qid, 'native_f1': n['f1'], 'head_f1': h['f1'], 'f1_delta': h['f1']-n['f1'],
                      'native_em': n['em'], 'head_em': h['em'], 'em_delta': h['em']-n['em'],
                      'support_f1_delta': h['support_f1']-n['support_f1'],
                      'native_answer': n['answer'], 'head_answer': h['answer'],
                      'native_errors': n['errors'], 'head_errors': h['errors'],
                      'tool_call_delta': h['tool_calls']-n['tool_calls'],
                      'first_divergence': divergence(n, h)})
    if status['phase'] == 'complete':
        assert len(pairs) == 100 and len(records) == status['completed'] == 200
        assert sum(e['completion_tokens'] for e in loaded.values()) == status['generated_tokens']
        assert sum(e['prompt_tokens'] for e in loaded.values()) == status['input_tokens']
    assert pairs
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, len(pairs), size=(5000, len(pairs)))
    paired = {}
    for key in ['f1_delta', 'em_delta', 'support_f1_delta']:
        values = np.array([p[key] for p in pairs])
        paired[key] = {'mean': float(values.mean()),
                       'bootstrap95_descriptive': np.quantile(values[indices].mean(1), [.025, .975]).tolist()}
    paired.update(n=len(pairs), f1_wins=sum(p['f1_delta'] > 0 for p in pairs),
                  f1_ties=sum(p['f1_delta'] == 0 for p in pairs), f1_losses=sum(p['f1_delta'] < 0 for p in pairs),
                  lost_exact=sum(p['native_em'] and not p['head_em'] for p in pairs),
                  gained_exact=sum(p['head_em'] and not p['native_em'] for p in pairs),
                  first_divergence_counts=dict(Counter(p['first_divergence']['kind'] for p in pairs)))
    n, h = groups['native'], groups['head_generate']
    ratios = {'observed_time_native_over_head': n['latency_total_seconds']/h['latency_total_seconds'],
              'generated_token_relative_change': h['completion_tokens']/n['completion_tokens']-1,
              'input_token_relative_change': h['prompt_tokens']/n['prompt_tokens']-1,
              'generated_plus_forced_prefix_native': n['completion_tokens']+n['forced_prefix_tokens'],
              'generated_plus_forced_prefix_head': h['completion_tokens']+h['forced_prefix_tokens']}
    conclusion = {'complete': status['phase'] == 'complete',
                  'head_point_f1_and_em_at_least_native': h['f1'] >= n['f1'] and h['em'] >= n['em'],
                  'quality_noninferiority_established': False, 'causal_speed_improvement_established': False,
                  'training_or_tuning_performed': False, 'next_automatic_training': False}
    report = {'manifest': manifest, 'status': status, 'groups': groups, 'paired': paired,
              'cost_observations': ratios, 'conclusion': conclusion, 'pairs': pairs, 'input_sha256': hashes,
              'notes': ['All native format-error episodes retained in primary metrics.',
                        'Full native baseline versus head+aligned cached generator: execution path and policy both differ.',
                        'Native argument_tokens=0 is uninstrumented combined generation, not zero actual argument generation.',
                        'Latency includes episode control, decoding, retrieval/query embedding and metering tokenization.',
                        'Model loading, document preembedding, scoring and writes excluded; corpus preparation logged separately.',
                        'Token/decoder counts do not prove FLOPs reduction; forced-prefix batch processing differs from sequential decoding.',
                        'New100 disjoint atomic families from available prior exposure and one another; selected population differs from full dev.',
                        'Observed shared-GPU timing, single rollout per question-policy; not controlled speedup.',
                        'No predefined noninferiority margin/power: bootstrap is uncertainty description, not noninferiority proof.']}
    (OUT/'frozen_head_generalization100_20261017.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'frozen_head_generalization100_20261017.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    lines = ['# Frozen 도구 head 새 2-hop100 평가', '',
             f"상태 `{status['phase']}`, paired {len(pairs)}문항. 모델·head 고정, 재학습·인자 선택 없음.", '',
             '| 지표 | Native | Head + 인자 생성 |', '|---|---:|---:|']
    for title, key, scale in [('답변 F1 (%)', 'f1', 100), ('EM (%)', 'em', 100), ('근거 F1 (%)', 'support_f1', 100),
                              ('평균 tool calls', 'tool_calls', 1), ('평균 interaction turns', 'interaction_turns', 1),
                              ('평균 생성 tokens', 'completion_tokens', 1), ('평균 forced-prefix tokens', 'forced_prefix_tokens', 1),
                              ('평균 입력 tokens', 'prompt_tokens', 1), ('평균 prefill 수', 'prefill_count', 1),
                              ('평균 decoder forward (산출)', 'decoder_forward_count_derived', 1),
                              ('평균 episode latency(s)', 'seconds', 1), ('latency 총합(s)', 'latency_total_seconds', 1),
                              ('오류 episode', 'error_episodes', 1)]:
        lines.append(f"| {title} | {n[key]*scale:.4f} | {h[key]*scale:.4f} |")
    lines += ['', f"F1 기준 head {paired['f1_wins']}승/{paired['f1_ties']}동률/{paired['f1_losses']}패. "
              f"Native exact 정답 상실{paired['lost_exact']} / 새 exact 획득{paired['gained_exact']}.",
              f"paired F1차이={paired['f1_delta']['mean']:.6f}, descriptive bootstrap95={paired['f1_delta']['bootstrap95_descriptive']}.",
              f"관측 시간 총합 Native/head={ratios['observed_time_native_over_head']:.3f}. 통제된speedup주장없음.", '',
              '## 해석과 한계', '',
              f"head F1·EM 점추정이 Native 이상인가: {conclusion['head_point_f1_and_em_at_least_native']}. "
              '이값은확인평가의점추정비교이며비열등성입증이아니다. 실패문항을보고이100문항에맞춰튜닝하지않는다. '
              '후속학습은train자료에서만하고새문항으로재평가해야한다.', '',
              '기존노출질문·분해하위질문family와겹치지않는100문항을outcome-blind hash순서로사전동결했다. '
              '새100내부의atomic family도불중복이다. 2-hop dev1252중1099eligible에서선별돼원dev전체의무작위표본과같지않다. '
              '모델pretraining노출까지배제한주장은아니다.', '',
              'Native오류도주분석에포함했다. head조건은도구선택뿐아니라aligned/cache실행경로도바뀌므로 '
              '전체차이를learnedhead선택효과하나로돌리지않는다. Native의인자토큰별도값0은미계측이다. '
              '생성+forcedprefix와입력토큰도함께보며출력감소를전체계산감소와동일시하지않는다.', '',
              'latency는실제episode제어·인자/답변생성·검색/queryembedding·계측용tokenization을포함한다. '
              '모델load/문서사전embedding/채점/파일저장제외. corpus준비는별도기록. '
              '순서교대와단일rollout의공유GPU관측값이며통제속도또는동일trajectory비교가아니다.', '',
              '개별답변·최초분기·qid별품질차이·source/inputhash는JSON,전episode수치는CSV에있다. '
              '최초분기는후보원인분류이며자동인과판정이나최적tool정답label이아니다. '
              '새head학습/3-hop확대는이번실행에포함하지않았다.']
    (OUT/'frozen_head_generalization100_20261017.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'groups': groups, 'paired': paired, 'cost_observations': ratios, 'conclusion': conclusion}, indent=2))


if __name__ == '__main__':
    main()
