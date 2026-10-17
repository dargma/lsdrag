"""CPU analysis of tested-branch oracle headroom; never train on oracle scores."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/contraction_probe_20261017'
OUT = REPO/'lab/reports/text_integrated_20261016'


def eligible(ep, base):
    return (not ep['errors'] and ep['answer_format_valid'] and ep['citation_observed_valid']
            and ep['em'] >= base['em'] and ep['f1']+1e-12 >= base['f1']
            and ep['observed_support_recall']+1e-12 >= base['observed_support_recall'])


def cost(ep):
    return (ep['full_tool_calls'], ep['full_decision_turns'], ep['generated_tokens'],
            ep['prefill_input_tokens']+ep['forced_prefix_tokens'])


def aggregate(es):
    keys = ['f1', 'em', 'support_f1', 'observed_support_recall', 'read_support_recall',
            'full_tool_calls', 'suffix_tool_calls', 'full_decision_turns', 'generated_tokens',
            'argument_tokens', 'forced_prefix_tokens', 'prefill_input_tokens', 'full_prefill_count', 'seconds',
            'decoder_forward_count_derived']
    return {'n': len(es), **{k: float(np.mean([e[k] for e in es])) for k in keys},
            'errors': sum(bool(e['errors']) for e in es),
            'invalid_observed_citation': sum(not e['citation_observed_valid'] for e in es)}


def main():
    status = json.loads((RUN/'status.json').read_text())
    manifest = json.loads((RUN/'manifest.json').read_text())
    rows, oracle, groups, inputs, quality_oracle = [], [], {}, {}, []
    for group in ['train', 'validation']:
        es = []
        for folder in sorted((RUN/'episodes'/group).glob('*')):
            branches = {}
            for path in sorted(folder.glob('*.json')):
                e = json.loads(path.read_text())
                e['decoder_forward_count_derived'] = sum(
                    max(0, c['usage'].get('generated_tokens', c['usage'].get('completion_tokens', 0))-1)
                    + int(c['usage'].get('forced_prefix_tokens', 0) > 0)
                    for c in e['reader_calls'] if c['usage'].get('generated_tokens', c['usage'].get('completion_tokens', 0)) > 0)
                assert e['full_tool_calls'] == len(e['trajectory'])
                assert e['suffix_tool_calls'] <= 2
                assert e['full_prefill_count'] == e['decision_turns']
                branches[e['branch']] = e
                es.append(e)
                inputs[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append({k: e[k] for k in ['group', 'qid', 'branch', 'f1', 'em', 'observed_support_recall',
                                              'read_support_recall', 'full_tool_calls', 'full_decision_turns',
                                              'generated_tokens', 'prefill_input_tokens', 'forced_prefix_tokens', 'seconds']})
            if 'head' not in branches:
                continue
            base = branches['head']
            candidates = [e for e in branches.values() if eligible(e, base)]
            best = min(candidates, key=cost) if candidates else None
            shorter = bool(best and (best['full_tool_calls'] < base['full_tool_calls'] or
                                    best['full_decision_turns'] < base['full_decision_turns']))
            alternates = [e for name, e in branches.items() if name.startswith('alt_') and eligible(e, base)
                          and (e['full_tool_calls'] < base['full_tool_calls'] or
                               e['full_decision_turns'] < base['full_decision_turns'])]
            final_suffices = 'FINAL' in branches and eligible(branches['FINAL'], base)
            early = [e for name, e in branches.items() if name in ['FINAL', 'head_B1'] and eligible(e, base)]
            early_best = min(early, key=cost) if early else None
            unique_nonterminal = [e for e in alternates if early_best is None
                                  or e['full_tool_calls'] < early_best['full_tool_calls']
                                  or e['full_decision_turns'] < early_best['full_decision_turns']]
            valid_branches = [e for e in branches.values() if not e['errors'] and e['answer_format_valid']
                              and e['citation_observed_valid']]
            quality_best = min(valid_branches, key=lambda e: (-e['f1'], -e['em'], *cost(e))) if valid_branches else None
            quality_oracle.append({'group': group, 'qid': base['qid'], 'baseline_f1': base['f1'],
                                   'best_tested_branch': quality_best['branch'] if quality_best else None,
                                   'best_tested_f1': quality_best['f1'] if quality_best else None,
                                   'quality_improvement_exists': bool(quality_best and quality_best['f1'] > base['f1'])})
            oracle.append({'group': group, 'qid': base['qid'], 'baseline_correct': bool(base['em']),
                           'baseline': {k: base[k] for k in ['answer', 'f1', 'em', 'observed_support_recall',
                                                           'full_tool_calls', 'full_decision_turns', 'generated_tokens',
                                                           'prefill_input_tokens', 'forced_prefix_tokens', 'full_prefill_count']},
                           'best': {k: best[k] for k in ['branch', 'answer', 'f1', 'em', 'observed_support_recall',
                                                       'full_tool_calls', 'full_decision_turns', 'generated_tokens',
                                                       'prefill_input_tokens', 'forced_prefix_tokens', 'full_prefill_count']} if best else None,
                           'shorter': shorter, 'immediate_final_suffices': final_suffices,
                           'nonterminal_shorter_branches': [e['branch'] for e in alternates],
                           'nonterminal_needed_beyond_immediate_final': bool(alternates) and not final_suffices,
                           'primary_early_stop_best_branch': early_best['branch'] if early_best else None,
                           'secondary_nonterminal_interaction_gain_beyond_primary_early_stop': bool(unique_nonterminal)})
        groups[group] = {name: aggregate([e for e in es if e['branch'] == name])
                         for name in sorted({e['branch'] for e in es})}
    assert len(rows) == status['completed']
    complete = status['phase'] == 'complete'
    if complete:
        assert len(oracle) == 32
    gates = {}
    for group in ['train', 'validation']:
        correct = [o for o in oracle if o['group'] == group and o['baseline_correct']]
        covered = [o for o in correct if o['best']]
        def reduction(key):
            return (1-sum(o['best'][key] for o in covered)/sum(o['baseline'][key] for o in covered)) if covered else None
        calls, turns = reduction('full_tool_calls'), reduction('full_decision_turns')
        nonterminal = sum(o['nonterminal_needed_beyond_immediate_final'] for o in correct)
        passing = (complete and len(correct) >= 5 and len(covered) == len(correct)
                   and sum(o['shorter'] for o in correct) >= 3 and max(calls or 0, turns or 0) >= .2)
        # Conservative component-wise Qwen proxy; no FLOPs/speed assertion.
        proxies = {k: reduction(k) for k in ['generated_tokens', 'prefill_input_tokens',
                                             'forced_prefix_tokens', 'full_prefill_count']}
        compute_direction = bool(covered) and all(v is not None and v >= -1e-12 for v in proxies.values())
        gates[group] = {'baseline_correct_questions': len(correct), 'oracle_quality_preserving_covered': len(covered),
                        'correct_questions_shorter': sum(o['shorter'] for o in correct),
                        'correct_questions_immediate_final_sufficient': sum(o['immediate_final_suffices'] for o in correct),
                        'nonterminal_shorter_not_explained_by_immediate_final': nonterminal,
                        'secondary_nonterminal_gain_beyond_immediate_and_B1_early_stop': sum(o['secondary_nonterminal_interaction_gain_beyond_primary_early_stop'] for o in correct),
                        'full_tool_call_reduction_correct_subset': calls,
                        'full_decision_turn_reduction_correct_subset': turns,
                        'qwen_component_reductions_correct_subset': proxies,
                        'qwen_component_direction_nonincreasing': compute_direction,
                        'oracle_quality_and_interaction_headroom_gate': passing,
                        'intermediate_contraction_investment_gate': passing and nonterminal >= 3 and compute_direction,
                        'statistics': 'exploratory investment thresholds only; oracle is privileged and validation is already exposed'}
    report = {'run': str(RUN), 'status': status, 'manifest': manifest, 'groups': groups,
              'oracle_per_question': oracle, 'quality_oracle_per_question': quality_oracle,
              'gates': gates, 'input_sha256': inputs,
              'training': False, 'learned_compression_policy': False, 'controlled_speed_claim': False,
              'limitations': ['one exposed state per question, not a new independent benchmark',
                              'bounded tested branches do not cover all possible action/query continuations',
                              'normal head and oracle both use suffix cap2; constrained diagnosis, not full native policy',
                              'native-name selector pays extra routing syntax and cache clone, not optimized native latency',
                              'oracle selects using gold QA/evidence; success does not prove inference-time learnability',
                              'prefix state is historical teacher; measured seconds are suffix-only',
                              'model/embedding load, scoring, saved corpus preparation and file writes are outside suffix time']}
    (OUT/'contraction_probe_20261017.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'contraction_probe_20261017.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Interaction 축소 가능성 제한 진단', '',
             f"실행 상태 `{status['phase']}`, 저장 probe {len(rows)}건. 새 head 학습 없음. "
             '같은 teacher 첫 tool 관측 이후 상태에서 실제 인자 생성·도구 실행·새 suffix를 비교했다. '
             '기존 train24/validation8이며 새로운 heldout 성능이 아니다.', '',
             '| 구간 | 조건 | n | F1 | EM | 전체 tool calls¹ | 전체 decision turns¹ | 생성 tokens | suffix latency(s) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, policies in groups.items():
        for name, s in policies.items():
            lines.append(f"| {group} | {name} | {s['n']} | {s['f1']:.4f} | {s['em']:.4f} | "
                         f"{s['full_tool_calls']:.2f} | {s['full_decision_turns']:.2f} | {s['generated_tokens']:.2f} | {s['seconds']:.3f} |")
    lines += ['', '¹공통prefix의 tool/decision1회를 포함한다. walltime과생성tokens는새suffix비용이며fulltrajectory end-to-end가아니다. '
              'B1은첫suffix tool후답변을별도로생성한고정예산baseline이다. native는최초선택만name경계decode이고후속정책은head이다. '
              '추가syntax처리/복제비용을포함해순수learnedselector속도기여로해석하지않는다. '
              'alt조건은합법성및head의최초선택에따라선별된서로다른질문subset이다. '
              '특히alt_semantic의높은평균을전체질문에서semantic만고르면되는증거로해석하지않는다.', '', '## Oracle headroom', '']
    for group, g in gates.items():
        lines += [f"- {group}: baseline정답{g['baseline_correct_questions']}질문, 품질보존더짧은질문{g['correct_questions_shorter']}, "
                  f"즉시FINAL충분{g['correct_questions_immediate_final_sufficient']}, "
                  f"즉시FINAL로설명되지않는nonterminal단축{g['nonterminal_shorter_not_explained_by_immediate_final']}. "
                  f"사후보조분석에서head_B1조기종료까지넘는nonterminal interaction추가이득{g['secondary_nonterminal_gain_beyond_immediate_and_B1_early_stop']}.",
                  f"  정답subset전체calls절감={g['full_tool_call_reduction_correct_subset']}, "
                  f"turn절감={g['full_decision_turn_reduction_correct_subset']}; "
                  f"quality/interaction여지gate={g['oracle_quality_and_interaction_headroom_gate']}, "
                  f"중간transition압축후속gate={g['intermediate_contraction_investment_gate']}."]
    lines += ['', 'Oracle는정답과관측근거를보고시험branch중고른특권적상한이다. baseline오답에서빨리오답을내는것은긍정적압축으로세지않았다. '
              '개별질문별oracle와cost성분은JSON,각probe는CSV와local-fast원본에보존했다. '
              'validation8은이미노출된확인진단이며gate재튜닝/통계적비열등성주장없음. '
              '최대2suffix tool계약과branch포괄성한계내결과다. 공유GPU관측latency이며controlledspeedup주장없음. '
              'decode forward수는각usage의생성길이와forcedprefix에서산출한값이며직접profiler계측이아니다(JSON에조건별평균저장). '
              'train alt_keyword_search 1건의FINAL512token상한truncation오류를그대로포함했다.', '',
              '## 해석', '',
              'train 정답5문항 중3문항에서 품질보존 단축이 가능했고 oracle 전체calls12→9(25%),turns17→14(17.65%)였다. 학습정책의 성능이 아니다. '
              'nonterminal 단축1문항은사전기준3문항에미달했다. 사후보조분석에서그1문항도head_B1조기종료로같은interaction절감이가능해, 중간transition대체의추가절감은확인되지않았다.', '',
              'validation8에서head F1.21875/EM.125,최초Native선택 F1.46875/EM.375. 같은관측에서read대신semantic/keyword를고른2문항이정답을회복했다. '
              '이는전체Native정책성능이아닌첫선택의진단이다. baseline정답1문항으로압축확인자료가부족하다. '
              '고정head_B1은trainF1.2676→.2525,validationF1.2188→.1827로낮아져oracle조기종료여지를곧바로고정step축소로배포하면안된다.', '',
              '권고: 현재도구head+LLM인자생성을유지하고도구선택오류·실패뒤재검색일반화를우선한다. 조기종료는작은가치예측후보로보존한다. '
              '현자료로중간stepdistillation학습이나새인자decoder를시작하지않는다.', '',
              '## 다음 단계', '',
              '이진단만으로새학습/RL/decoder를자동실행하지않는다. nonterminal gate가안되면중간stepdistillation학습을시작하지않고, '
              'FINAL-only이득은early stopping으로구분한다. 같은도구head+LLM인자생성기준선과원래일반화계획은보존한다.']
    (OUT/'contraction_probe_20261017.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': status, 'gates': gates}, indent=2))


if __name__ == '__main__':
    main()
