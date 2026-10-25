"""POC: action/stop headroom from saved 9B all-action branches (train48, suffix = bootstrap head). CPU only."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ACTIONS = ['keyword_search', 'semantic_search', 'read_chunk', 'FINAL']


def main():
    run = Path(sys.argv[1])
    branches = defaultdict(dict)
    base = {}
    for f in (run/'episodes').glob('*.json'):
        ep = json.loads(f.read_text())
        m = re.match(r'(.+)_t(\d+)_a(\d)_r(\d+)\.json$', f.name)
        if m:
            branches[(m[1], int(m[2]))][int(m[3])] = ep
        elif f.name.endswith('_collector.json'):
            base[f.name[:-len('_collector.json')]] = ep
    states = []
    for (qid, t), acts in sorted(branches.items()):
        col = base[qid]
        chosen = col['decisions'][t]['action'] if t < len(col['decisions']) else None
        out = {a: dict(f1=e['f1'], em=e['em'], prompt=e['prompt_tokens'], gen=e['generated_tokens'], calls=e['tool_calls'])
               for a, e in acts.items()}
        states.append(dict(qid=qid, t=t, chosen=chosen, out=out))
    # Sanity: the branch that repeats the chosen action should reproduce the collector suffix.
    reproduced = [s for s in states if s['chosen'] in s['out']]
    same = sum(abs(s['out'][s['chosen']]['f1']-base[s['qid']]['f1']) < 1e-9 for s in reproduced)
    multi = [s for s in states if len(s['out']) >= 2]
    sensitive = [s for s in multi if max(o['f1'] for o in s['out'].values()) > min(o['f1'] for o in s['out'].values())]
    em_sensitive = [s for s in multi if len({o['em'] for o in s['out'].values()}) > 1]
    # One-deviation oracle per question: best F1 achievable by changing one action at one visited state.
    per_q = defaultdict(list)
    for s in states:
        per_q[s['qid']].append(s)
    q_rows = []
    for qid, ss in per_q.items():
        b = base[qid]
        best_f1 = max([b['f1']]+[o['f1'] for s in ss for o in s['out'].values()])
        best_em = max([b['em']]+[o['em'] for s in ss for o in s['out'].values()])
        # Stop headroom: earliest state where FINAL keeps base EM and F1; input tokens = prefix (from collector) + branch.
        prefix = []
        acc = 0
        calls = b['reader_calls']
        i = 0
        for d in b['decisions']:
            acc += sum(u.get('prompt_tokens', 0) for u in calls[i:i+2])
            i += 2
            prefix.append(acc)
        stop = None
        for s in sorted(ss, key=lambda s: s['t']):
            o = s['out'].get(3)
            if o and o['em'] >= b['em'] and o['f1'] >= b['f1']-1e-9:
                before = prefix[s['t']-1] if s['t'] > 0 else 0
                stop = dict(t=s['t'], inputs=before+o['prompt'])
                break
        q_rows.append(dict(qid=qid, base_f1=b['f1'], base_em=b['em'], base_inputs=b['prompt_tokens'], base_calls=b['tool_calls'],
                           best_one_dev_f1=best_f1, best_one_dev_em=best_em, quality_preserving_stop=stop))
    n = len(q_rows)
    stop_inputs = sum((r['quality_preserving_stop'] or {}).get('inputs', r['base_inputs']) for r in q_rows)
    summary = dict(
        questions=n, states=len(states), states_with_2plus_actions=len(multi),
        chosen_branch_reproduces_collector=f'{same}/{len(reproduced)}',
        f1_sensitive_states=len(sensitive), em_sensitive_states=len(em_sensitive),
        base_mean_f1=sum(r['base_f1'] for r in q_rows)/n, base_em=sum(r['base_em'] for r in q_rows),
        one_deviation_oracle_mean_f1=sum(r['best_one_dev_f1'] for r in q_rows)/n,
        one_deviation_oracle_em=sum(r['best_one_dev_em'] for r in q_rows),
        base_mean_inputs=sum(r['base_inputs'] for r in q_rows)/n,
        quality_preserving_stop_mean_inputs=stop_inputs/n,
        quality_preserving_stop_input_saving=1-stop_inputs/sum(r['base_inputs'] for r in q_rows),
        base_mean_calls=sum(r['base_calls'] for r in q_rows)/n,
        caveat='Oracle = max over deterministic greedy branches (one deviation, then bootstrap-head suffix); upward-biased upper bound. Train48 only; suffix policy is bootstrap, not value_aux.')
    (run/'poc_branch_headroom.json').write_text(json.dumps(dict(summary=summary, questions=q_rows, states=states), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
