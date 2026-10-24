"""Select at most two learned candidates using validation only; freeze test spec."""
import json
from pathlib import Path
from run_decision_policy_pilot import ROOT, REPO, save, sha

def read(path):return json.loads(path.read_text())

def main():
    plan=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json')
    assert sha(REPO/'scripts/select_decision_shared_final.py')==plan['selector_sha256']
    split=read(REPO/'lab/registry/decision_shared_outcomes192_20261023/manifest.json')
    names=['decision_shared_validation64_20261023','decision_shared_structure_validation64_20261023']
    policies={};values={};provenance={};common=None
    for name in names:
        run=ROOT/'results'/name
        assert read(run/'status.json')['phase']=='complete'
        spec=read(run/'spec.json')
        assert spec['split']=='validation' and spec['qids']==split['validation']
        identity={k:spec[k] for k in ['qids','model','corpus','document_sha256','vectors_sha256','source_sha256']}
        if common is None:common=identity
        else:assert common==identity, 'Comparisons must share exact corpus/model/runtime sources'
        summary=read(run/'summary.json')
        for key,value in summary['policies'].items():
            assert key not in policies
            policies[key]=spec['policies'][key];values[key]=value
            provenance[key]=dict(run=str(run),spec_sha256=sha(run/'spec.json'),summary_sha256=sha(run/'summary.json'))
    base=values['shared_grpo'];scores={}
    for name in plan['eligible']:
        v=values[name];delta=v['f1']-base['f1'];reduction=1-v['generated_tokens']/base['generated_tokens']
        scores[name]=dict(f1_delta=delta,generated_reduction=reduction,joint_score=min(delta/.05,reduction/.30),utility=v['f1']-.03*v['cost_units'])
    ranked=sorted(plan['eligible'],key=lambda n:(scores[n]['joint_score'],scores[n]['utility'],n),reverse=True)
    primary=ranked[0]
    secondary=next(n for n in ranked[1:] if policies[n]['kind']!=policies[primary]['kind'])
    chosen=[primary,secondary]
    policies['closed_book']={'kind':'final_only'}
    conditions=['shared_grpo','shared_bc','semantic_once','closed_book','native_fast',*chosen,'multi_mlp_bc']
    assert len(set(conditions))==len(conditions)
    for name in conditions:
        if 'checkpoint' in policies[name]:assert sha(Path(policies[name]['checkpoint']))==policies[name]['sha256']
    assert not (ROOT/'results/decision_shared_test160_20261023').exists()
    final=read(ROOT/'results'/names[0]/'spec.json')
    final.update(run_name='decision_shared_test160_20261023',split='test',qids=split['test'],policies={n:policies[n] for n in conditions},selection='Two learned candidates selected from validation only by predeclared joint score, with distinct architecture; fixed baselines and matched multi-MLP warmstart retained.',max_seconds=3600,max_generated=4000000,max_inputs=160000000)
    final['source_sha256']={p:sha(REPO/p) for p in final['source_sha256']}
    final['source_sha256']['src/lsdrag/decision_closed_book.py']=sha(REPO/'src/lsdrag/decision_closed_book.py')
    folder=REPO/'lab/registry/decision_shared_final_20261023'
    assert not (folder/'selection.json').exists() and not (folder/'spec.json').exists()
    save(folder/'selection.json',dict(primary=primary,secondary=secondary,scores=scores,all_validation=values,provenance=provenance,plan_sha256=sha(folder/'selection_plan.json'),test_outcomes_observed=False,warning='Validation selection is not target attainment. Final target is relative to shared GRPO; usefulness must also be assessed against native and semantic-once baselines.'))
    save(folder/'spec.json',final)
    print(json.dumps(dict(primary=primary,secondary=secondary,conditions=conditions,test_questions=len(final['qids'])),indent=2))

if __name__=='__main__':main()
