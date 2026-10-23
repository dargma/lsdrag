"""Freeze same-initialization learning inputs after the teacher/BC stage."""
import json
from pathlib import Path
from run_decision_policy_pilot import ROOT, REPO, save, sha

def read(path):
    return json.loads(path.read_text())

def main():
    initialization = ROOT/'results/decision_shared_initialization_20261023'
    assert read(initialization/'status.json')['phase']=='complete'
    assert read(ROOT/'results/decision_shared_initialization_validation16_20261023/status.json')['phase']=='complete'
    init = read(initialization/'manifest.json')
    split = read(ROOT/'results/decision_state_policy_iteration_20261023/manifest.json')
    assert init['train'] == split['train']
    reference = {k:init[k] for k in ['kind','checkpoint','sha256']}
    corpus = ROOT/'corpora/musique_shared_20261023'
    cm = read(corpus/'manifest.json')
    assert cm['embeddings_complete']
    inputs = [Path(init['checkpoint']), corpus/'documents.jsonl', corpus/'vectors.npy']
    inputs += [ROOT/f'source/musique/musique_ans_v1.0_{s}.jsonl' for s in ['train','dev']]
    common_sources = ['src/lsdrag/decision_policy.py','src/lsdrag/decision_policy_recovery.py','src/lsdrag/decision_policy_followup.py','src/lsdrag/text_decision_benchmark.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','src/lsdrag/decision_grpo.py','src/lsdrag/decision_state_learning.py','prompts/text_rag_integrated_v5.txt','scripts/fit_decision_state_heads.py']
    common = dict(train=split['train'],validation=split['validation'],test=split['test'],model=split['model'],corpus=str(corpus),reference=init['checkpoint'],reference_spec=reference,reference_name='shared_bc',input_sha256={str(p):sha(p) for p in inputs},max_generated=1800000,max_inputs=100000000,target=dict(f1_delta=.05,generated_ratio=.70),note='Shared-corpus restart, original eight-hour deadline elapsed before explicit user resumption. Per-stage bounded budgets below. Same train questions/initial checkpoint/actor trainability, but trajectory and token budgets differ; not a strictly compute-matched causal comparison.')
    stages = [
        ('decision_shared_grpo192_20261023','scripts/run_decision_shared_grpo.py',dict(seed=20261052,iterations=2,repeats=4,budget_episodes=1536,max_seconds=3600,lr=1e-4)),
        ('decision_shared_outcomes192_20261023','scripts/run_decision_shared_outcomes.py',dict(seed=20261053,states_per_question=2,repeats_per_action=2,budget_episodes=3264,max_seconds=7200,suffix_policy='Frozen newly trained shared_bc, stochastic epsilon0; first action is intervention. Never used as on-policy GRPO samples.')),
    ]
    for name, script, extra in stages:
        path = REPO/'lab/registry'/name/'manifest.json'
        assert not path.exists()
        sources = common_sources+[script]
        if name.startswith('decision_shared_outcomes'):
            sources += ['scripts/fit_decision_shared_outcomes.py']
        save(path, common|extra|dict(source_sha256={p:sha(REPO/p) for p in sources}))
    print('Frozen train192, validation64, untouched test160; GRPO1536 and all-action<=3264 training episodes.')

if __name__=='__main__':
    main()
