"""Guard final evaluation scope without reading any real held-out outcomes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import pytest

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

@pytest.fixture
def selection_environment(tmp_path,monkeypatch):
    real_repo=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(real_repo/'scripts'))
    module=importlib.import_module('select_decision_shared_final')
    repo=tmp_path/'repo';root=tmp_path/'runtime'
    monkeypatch.setattr(module,'REPO',repo);monkeypatch.setattr(module,'ROOT',root)
    script=repo/'scripts/select_decision_shared_final.py';script.parent.mkdir(parents=True)
    script.write_bytes((real_repo/'scripts/select_decision_shared_final.py').read_bytes())
    folder=repo/'lab/registry/decision_shared_final_20261023'
    questions=[f'q{i}' for i in range(160)];validation=[f'v{i}' for i in range(64)]
    put(folder/'selection_plan.json',dict(selector_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),eligible=['policy_iteration_cost003','policy_iteration_cost006','multi_mlp_outcome'],test_qids=questions[:80],reserved_qids=questions[80:]))
    put(repo/'lab/registry/decision_shared_outcomes192_20261023/manifest.json',dict(validation=validation,test=questions))
    values={'shared_grpo':(.5,100.),'shared_bc':(.4,100.),'native_fast':(.5,90.),'semantic_once':(.4,50.),'policy_iteration_cost003':(.6,100.),'policy_iteration_cost006':(.56,70.),'multi_mlp_bc':(.5,100.),'multi_mlp_outcome':(.7,95.)}
    for name,conditions in [('decision_shared_validation64_20261023',list(values)[:6]),('decision_shared_structure_validation64_20261023',list(values)[6:])]:
        run=root/'results'/name;policies={}
        for key in conditions:
            if key in ['native_fast','semantic_once']:
                policies[key]={'kind':'native' if key=='native_fast' else 'rule'}
            else:
                checkpoint=root/(key+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(key.encode())
                policies[key]=dict(kind='structure' if key.startswith('multi') else 'original',checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        put(run/'status.json',dict(phase='complete'))
        put(run/'spec.json',dict(split='validation',qids=validation,model='fixture',corpus='fixture',document_sha256='fixture',vectors_sha256='fixture',source_sha256={},policies=policies))
        put(run/'summary.json',dict(policies={key:dict(f1=values[key][0],generated_tokens=values[key][1],cost_units=2.) for key in conditions}))
    return module,folder,questions

def test_joint_tradeoff_selects_one_candidate_and_preserves_reserved_questions(selection_environment):
    module,folder,questions=selection_environment
    module.main()
    selection=json.loads((folder/'selection.json').read_text());spec=json.loads((folder/'spec.json').read_text())
    # Highest F1 alone is not the declared objective; .56/70 beats .70/95 here.
    assert selection['primary']=='policy_iteration_cost006'
    assert list(spec['policies'])==['shared_grpo','policy_iteration_cost006','native_fast','semantic_once']
    assert spec['qids']==questions[:80] and not set(spec['qids'])&set(questions[80:])
    assert spec['run_name']=='decision_shared_test80_20261023'
    assert selection['test_outcomes_observed'] is False

def test_reserved_question_overlap_is_rejected_before_writing_final_spec(selection_environment):
    module,folder,questions=selection_environment
    path=folder/'selection_plan.json';plan=json.loads(path.read_text());plan['test_qids'][0]=questions[80];put(path,plan)
    with pytest.raises(AssertionError):module.main()
    assert not (folder/'spec.json').exists()
