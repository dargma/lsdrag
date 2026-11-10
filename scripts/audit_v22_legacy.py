"""Compare restored historical i0 episodes against the complete S0 reproduction."""
import json
from common import ROOT, save, sha


def main():
    current = ROOT/'results/v22_legacy_dev192'
    prior = ROOT/'legacy_evidence/mh_native_A1/eval/i0'
    assert json.loads((current/'status.json').read_text())['phase'] == 'complete'
    old = {p.stem: json.loads(p.read_text()) for p in prior.glob('*.json') if p.stem != 'summary'}
    new = {r['qid']: r for p in (current/'episodes').glob('*.json') for r in [json.loads(p.read_text())]}
    assert set(old) == set(new) and len(new) == 192
    fields = ['f1', 'em', 'tool_calls', 'generated_tokens', 'prompt_tokens']
    means = {k: {'old': sum(r[k] for r in old.values())/192,
                 'new': sum(r[k] for r in new.values())/192} for k in fields}
    for v in means.values():
        v['delta'] = v['new']-v['old']
    mismatches = []
    for qid in sorted(new):
        a, b = old[qid], new[qid]
        trajectory_a = [(t['tool_name'], t['arguments'], t['result']) for t in a['trajectory']]
        trajectory_b = [(t['tool_name'], t['arguments'], t['result']) for t in b['trajectory']]
        if a['answer'] != b['answer'] or trajectory_a != trajectory_b:
            mismatches.append(dict(qid=qid, old_answer=a['answer'], new_answer=b['answer'],
                                   old_actions=[t[0] for t in trajectory_a], new_actions=[t[0] for t in trajectory_b],
                                   old_f1=a['f1'], new_f1=b['f1']))
    result = dict(questions=192, means=means, exact_answer_and_trajectory_matches=192-len(mismatches),
                  mismatches=mismatches, new_errors=sum(bool(r['errors']) for r in new.values()),
                  source_spec_sha256=sha(current/'spec.json'),
                  exact_reproduction=not mismatches and all(abs(v['delta']) < 1e-12 for v in means.values()))
    save(ROOT/'specs/v22/legacy_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
