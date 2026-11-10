"""CPU-only scoring of frozen group8 rollouts. Does not claim RL-optimal weights."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from common import save, sha, REPO
from lsdrag.fullft import v22_step_rewards, v22_group_advantages


def question_split(qids):
    calibration, audit = [], []
    for hop in ('2hop', '3hop', '4hop'):
        group = sorted((q for q in qids if q.startswith(hop)),
                       key=lambda q: hashlib.sha256(f'20261110:reward:{q}'.encode()).hexdigest())
        if len(group) != 8:
            raise ValueError('Exactly 8 pilot questions per hop required')
        calibration.extend(group[:5]); audit.extend(group[5:])
    return calibration, audit


def score_groups(groups, qids, supports, process, call_weight, token_weight):
    result = {}
    for arm in ('head', 'native'):
        signal_groups = diverse_groups = direct_groups = rank_correct = rank_total = 0
        zeros = units = 0
        advantage_sizes = []
        details = []
        for qid in qids:
            episodes = groups[(arm, qid)]
            if len(episodes) != 8:
                raise ValueError('Missing/duplicated pilot group8')
            gold = set(supports[qid])
            returns = [sum(v22_step_rewards(e, gold, process, call_weight, token_weight)) for e in episodes]
            advantages = v22_group_advantages(episodes, gold, process, call_weight, token_weight)
            decisions = [a for ep, row in zip(episodes, advantages)
                         for a, d in zip(row, ep['decisions']) if not d['forced']]
            zeros += sum(abs(a) <= 1e-8 for a in decisions); units += len(decisions)
            advantage_sizes.extend(abs(a) for a in decisions)
            signal_groups += max(returns)-min(returns) > 1e-8
            patterns = {tuple(d['action'] for d in ep['decisions'] if not d['forced']) for ep in episodes}
            diverse_groups += len(patterns) >= 2
            correct = [i for i, e in enumerate(episodes) if e['em'] == 1 and not e['errors']]
            direct = len(correct) >= 2 and len({episodes[i]['tool_calls'] for i in correct}) >= 2
            direct_groups += direct
            if direct:
                best_cost = min((episodes[i]['tool_calls'], episodes[i]['generated_tokens']) for i in correct)
                top = max(returns)
                winners = [i for i, v in enumerate(returns) if abs(v-top) <= 1e-8]
                agrees = all(i in correct and (episodes[i]['tool_calls'], episodes[i]['generated_tokens']) == best_cost
                             for i in winners)
                rank_correct += agrees; rank_total += 1
            details.append(dict(qid=qid, correct_count=len(correct), direct_call_signal=direct,
                                unique_tool_sequences=len(patterns), returns=returns))
        result[arm] = dict(questions=len(qids), reward_varying_groups=signal_groups,
                           reward_varying_fraction=signal_groups/len(qids),
                           diverse_tool_groups=diverse_groups, diverse_tool_fraction=diverse_groups/len(qids),
                           zero_adv_decisions=zeros, decisions=units, zero_adv_fraction=zeros/units if units else 1.,
                           direct_call_groups=direct_groups, direct_call_fraction=direct_groups/len(qids),
                           ranking_groups=rank_total, ranking_agreement=rank_correct/rank_total if rank_total else None,
                           mean_abs_advantage=sum(advantage_sizes)/max(1, len(advantage_sizes)), details=details)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True); args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    split = json.loads((REPO/'configs/split_mixhop.json').read_text())
    assert len(spec['qids']) == len(set(spec['qids'])) == 24
    assert set(spec['qids']) <= set(split['train'])
    groups = defaultdict(list)
    for path in spec['episode_files']:
        if sha(Path(path)) != spec['episode_sha256'][path]:
            raise ValueError('Pilot evidence changed after freeze')
        ep = json.loads(Path(path).read_text())
        assert ep['qid'] in spec['qids']
        groups[(ep['arm'], ep['qid'])].append(ep)
    cal, audit = question_split(spec['qids'])
    assert cal == spec['calibration_qids'] and audit == spec['audit_qids']
    assert len(groups) == 48
    for episodes in groups.values():
        assert {ep['rep'] for ep in episodes} == set(range(8)), 'Require eight distinct rollout repetitions'
        episodes.sort(key=lambda ep: ep['rep'])
    supports = spec['train_support_ids']
    default = score_groups(groups, spec['qids'], supports, False, .15, .10)
    process = any(x['zero_adv_fraction'] >= .20 for x in default.values())
    candidates = []
    for cw in (.05, .10, .15, .20):
        for tw in (.05, .10, .15):
            metrics = score_groups(groups, cal, supports, process, cw, tw)
            agreements = [r['ranking_agreement'] for r in metrics.values()]
            rank = min(agreements) if all(v is not None for v in agreements) else -1.
            candidates.append(dict(call_weight=cw, token_weight=tw, minimum_arm_ranking_agreement=rank,
                                   calibration=metrics))
    # OBS_02: lexicographic call preference biases this score toward the largest
    # coefficient ratio. Report it descriptively; never use it to select weights.
    cw, tw = .15, .10
    selected = next(r for r in candidates if r['call_weight'] == cw and r['token_weight'] == tw)
    confirmation = score_groups(groups, audit, supports, process, cw, tw)
    all_metrics = score_groups(groups, spec['qids'], supports, process, cw, tw)
    stop = any(m['zero_adv_fraction'] >= .2 or m['reward_varying_fraction'] < .5
               or m['diverse_tool_fraction'] < .5 for m in all_metrics.values())
    needs_user = any(m['direct_call_fraction'] < .1 for m in all_metrics.values())
    result = dict(phase='gate_failed' if stop else 'user_confirmation_required' if needs_user else 'passed',
                  process_reward=process, call_weight=cw, token_weight=tw, default_terminal_diagnostics=default,
                  candidates=candidates, audit=confirmation, all_pilot=all_metrics,
                  correct_pair_diagnostic_available=selected['minimum_arm_ranking_agreement'] >= 0,
                  coefficient_selection='unchanged_default; grid is diagnostic only (PLAN rev6)',
                  spec_sha256=sha(Path(args.spec)), note='Offline ranking/signals only; not trained-policy quality or a proven optimal reward')
    save(Path(spec['output']), result)
    print(json.dumps({k: result[k] for k in ('phase','process_reward','call_weight','token_weight','correct_pair_diagnostic_available')}, indent=2))


if __name__ == '__main__':
    main()
