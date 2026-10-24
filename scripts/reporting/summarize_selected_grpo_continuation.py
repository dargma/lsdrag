"""Compare paired development results with the previously measured GRPO reference."""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/content/local_fast/lsdrag-text-v1/results')
NEW = 'decision_selected_head_grpo_validation64_20261024_r1'


def read(path):
    return json.loads(path.read_text())


def main():
    names = [NEW, 'decision_shared_validation64_20261023',
             'decision_shared_structure_validation64_20261023']
    runs = {}
    for name in names:
        run = ROOT/name
        assert read(run/'status.json')['phase'] == 'complete'
        spec = read(run/'spec.json')
        assert spec['split'] == 'validation'
        episodes = [read(p) for p in (run/'episodes').glob('*.json')]
        cells = {(e['policy'], e['qid']): e for e in episodes}
        assert len(cells) == len(spec['qids'])*len(spec['policies']) == len(episodes)
        runs[name] = (spec, cells)
    current, cells = runs[NEW]
    qids = current['qids']
    drift = {}
    for name, (spec, _) in runs.items():
        for key in ['qids', 'model', 'corpus', 'document_sha256', 'vectors_sha256']:
            assert spec[key] == current[key], (name, key)
        assert spec['source_sha256'].keys() == current['source_sha256'].keys()
        drift[name] = [p for p in spec['source_sha256']
                       if spec['source_sha256'][p] != current['source_sha256'][p]]
        assert set(drift[name]) <= {'src/lsdrag/decision_policy.py'}
    previous, old_cells = runs[names[2]]
    assert current['policies']['multi_mlp_outcome'] == previous['policies']['multi_mlp_outcome']
    fields = ['answer', 'support_ids', 'f1', 'generated_tokens', 'prompt_tokens', 'tool_calls']
    matches = sum(all(cells['multi_mlp_outcome', q][k] == old_cells['multi_mlp_outcome', q][k]
                      for k in fields) and
                  [d['action'] for d in cells['multi_mlp_outcome', q]['decisions']] ==
                  [d['action'] for d in old_cells['multi_mlp_outcome', q]['decisions']]
                  for q in qids)
    plan = read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json')
    assert set(qids).isdisjoint(plan['test_qids']+plan['reserved_qids'])
    policy_runs = {'multi_mlp_outcome': NEW, 'multi_mlp_outcome_grpo': NEW,
                   'shared_grpo': names[1], 'native_fast': names[1], 'semantic_once': names[1]}
    vectors = {}
    means = {}
    for policy, name in policy_runs.items():
        es = [runs[name][1][policy, q] for q in qids]
        vectors[policy] = {k: np.array([e[k] for e in es])
                           for k in ['f1', 'generated_tokens', 'prompt_tokens', 'tool_calls']}
        means[policy] = {k: float(v.mean()) for k, v in vectors[policy].items()}
        means[policy]['errors'] = sum(bool(e['errors']) for e in es)
    rng = np.random.default_rng(20261046)
    draws = rng.integers(len(qids), size=(10000, len(qids)))
    comparisons = {}
    new = vectors['multi_mlp_outcome_grpo']
    for reference in ['multi_mlp_outcome', 'shared_grpo', 'native_fast', 'semantic_once']:
        old = vectors[reference]
        delta = new['f1']-old['f1']
        ratio = float(new['generated_tokens'].mean()/old['generated_tokens'].mean())
        comparisons[reference] = dict(f1_delta=float(delta.mean()),
            f1_ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
            generated_ratio=ratio,
            generated_ratio_ci95=np.quantile(new['generated_tokens'][draws].mean(1)/
                                              old['generated_tokens'][draws].mean(1), [.025, .975]).tolist(),
            point_target_met=bool(delta.mean() >= .05 and ratio <= .70))
    result = dict(split='validation', questions=len(qids), policies=means,
        comparisons=comparisons, baseline_replay_matches=matches,
        baseline_replay_fields=fields+['action_sequence'], policy_run_provenance=policy_runs,
        source_hash_changes=drift, final80_and_reserved80_unused=True,
        note='Exposed validation64 only. Earlier GRPO/native/rule results are cached paired references. '
             'Runtime change is optional state-storage dtype with legacy BF16 default; '
             'before-head replay equality is reported, not assumed. Marginal95% question bootstrap; '
             'single training seed, no multiplicity correction, no fresh-test target claim.')
    out = REPO/'lab/reports'/NEW
    out.mkdir(parents=True, exist_ok=True)
    (out/'cross_run_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
