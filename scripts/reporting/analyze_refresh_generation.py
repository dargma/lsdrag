"""Train-only generation accounting; no policy fitting or evaluation access."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/content/local_fast/lsdrag-text-v1/results/decision_current_policy_refresh48_20261024')
OUT = Path('lab/reports/decision_current_policy_refresh48_20261024')
ACTIONS = ['keyword_search', 'semantic_search', 'read_chunk', 'FINAL']

def main():
    totals = collections.Counter()
    actions = {a: collections.Counter() for a in ACTIONS}
    files = sorted((ROOT / 'episodes').glob('*_collector.json'))
    provenance = {}
    for path in files:
        raw = path.read_bytes()
        provenance[path.name] = hashlib.sha256(raw).hexdigest()
        ep = json.loads(raw)
        calls = ep['reader_calls']
        decisions = ep['decisions']
        assert len(calls) == 2 * len(decisions)
        measured = 0
        for i, decision in enumerate(decisions):
            prefill, generation = calls[2*i:2*i+2]
            assert prefill.get('pack_slots') == 37
            assert prefill.get('generated_tokens', 0) == 0
            n = generation['generated_tokens']
            measured += n
            counter = actions[ACTIONS[decision['action']]]
            counter['decisions'] += 1
            counter['generated'] += n
            counter['forced'] += int(decision['forced'])
        assert measured == ep['generated_tokens']
        seen = set()
        for step in ep['trajectory']:
            identity = json.dumps([step['tool_name'], step['arguments']], sort_keys=True)
            totals['exact_repeated_tool_calls'] += int(identity in seen)
            seen.add(identity)
        totals['questions'] += 1
        totals['generated'] += measured
        totals['tool_calls'] += ep['tool_calls']
        totals['error_episodes'] += bool(ep['errors'])
    assert totals['questions'] == 48
    result = {'scope': '48 stochastic training collector episodes only; intervention suffixes excluded',
              'totals': dict(totals), 'actions': {k: dict(v) for k,v in actions.items()},
              'source_sha256': provenance,
              'limitations': ['Accounting only, not measured savings.',
                'Repeated identical calls can return useful context; removal needs evaluation.',
                'All read generation is an optimistic upper bound for read-argument savings, not an achievable estimate.',
                'No validation or test access; stochastic training sample is not validation performance.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'generation_accounting.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Train48 generation accounting', '', result['scope'], '',
             '| Action | Decisions | Generated tokens | Share |', '|---|---:|---:|---:|']
    for a,c in actions.items():
        lines.append(f"| {a} | {c['decisions']} | {c['generated']} | {100*c['generated']/totals['generated']:.2f}% |")
    lines += ['', json.dumps(dict(totals)), '', *result['limitations']]
    (OUT/'generation_accounting.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, indent=2))

if __name__ == '__main__':
    main()
