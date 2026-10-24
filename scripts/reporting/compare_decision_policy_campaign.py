"""Paired exposed-validation comparison; never declares research goal achieved."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
REPO=Path(__file__).resolve().parents[2]
BASE='decision_shared_validation64_20261023'
PRIOR='decision_selected_head_grpo_validation64_20261024_r1'

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load(name):
    path=ROOT/name
    assert read(path/'status.json')['phase']=='complete',name
    spec=read(path/'spec.json');assert spec['split']=='validation'
    assert len(set(spec['qids']))==64==len(spec['qids'])
    eps=[read(p) for p in (path/'episodes').glob('*.json')]
    cells={(e['policy'],e['qid']):e for e in eps}
    assert len(cells)==len(eps)==len(spec['qids'])*len(spec['policies'])
    assert set(cells)=={(p,q) for p in spec['policies'] for q in spec['qids']}
    return spec,cells

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    assert Path(args.run).name==args.run
    names=list(dict.fromkeys([BASE,PRIOR,args.run]));runs={n:load(n) for n in names}
    base=runs[BASE][0];current=runs[args.run][0];qids=base['qids']
    assert base['policies']['shared_grpo']['sha256']=='59a0f76672a3411abaa53ab8fd8f40e0c8ddc027d26d566fea25a7187e4421ef'
    selection=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json')
    assert not set(qids)&set(selection['test_qids']+selection['reserved_qids'])
    source_drift={}
    for name,(spec,_) in runs.items():
        for k in ['qids','model','corpus','document_sha256','vectors_sha256']:assert spec[k]==base[k],(name,k)
        shared=base['source_sha256'].keys()&spec['source_sha256'].keys()
        changed=[k for k in shared if base['source_sha256'][k]!=spec['source_sha256'][k]]
        assert set(changed)<={'src/lsdrag/decision_policy.py'},(name,changed)
        source_drift[name]={'changed':changed,'added':sorted(spec['source_sha256'].keys()-base['source_sha256'].keys())}
    references={'basic_grpo':(BASE,'shared_grpo'),'native':(BASE,'native_fast'),
                'selected_before':(PRIOR,'multi_mlp_outcome'),'selected_ungated_grpo':(PRIOR,'multi_mlp_outcome_grpo')}
    candidates={p:(args.run,p) for p in current['policies']}
    all_specs=references|candidates;vectors={};summary={}
    for label,(run,policy) in all_specs.items():
        es=[runs[run][1][policy,q] for q in qids]
        vectors[label]={k:np.array([e[k] for e in es]) for k in ['f1','generated_tokens','prompt_tokens','tool_calls']}
        summary[label]={k:float(v.mean()) for k,v in vectors[label].items()}
        summary[label].update(errors=sum(bool(e['errors']) for e in es),zero_tool_episodes=sum(e['tool_calls']==0 for e in es),empty_answers=sum(not e['answer'].strip() for e in es))
    draws=np.random.default_rng(20261046).integers(64,size=(10000,64))
    comparisons={}
    pairs=[(p,r) for p in candidates for r in references]
    pairs += [(p,'mlp_control') for p in candidates if 'mlp_control' in candidates and p!='mlp_control']
    for body in ['mlp','attention']:
        if body+'_stop_value' in candidates and body+'_stop' in candidates:pairs.append((body+'_stop_value',body+'_stop'))
    for candidate,reference in pairs:
        new,old=vectors[candidate],vectors[reference]
        delta=new['f1']-old['f1'];ratio=float(new['generated_tokens'].mean()/old['generated_tokens'].mean())
        comparisons[candidate+'__vs__'+reference]=dict(f1_delta=float(delta.mean()),f1_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),generated_ratio=ratio,generated_ratio_ci95=np.quantile(new['generated_tokens'][draws].mean(1)/old['generated_tokens'][draws].mean(1),[.025,.975]).tolist(),wins=int((delta>1e-8).sum()),losses=int((delta< -1e-8).sum()),development_point_threshold_met=bool(delta.mean()>=.05 and ratio<=.7))
    out=REPO/'lab/reports'/args.run;out.mkdir(parents=True,exist_ok=True)
    result=dict(split='exposed_validation64',goal_achieved=False,policies=summary,comparisons=comparisons,provenance=all_specs,source_drift=source_drift,spec_sha256={n:digest(ROOT/n/'spec.json') for n in names},note='Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.')
    (out/'campaign_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Development comparison','', '|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|','|---|---:|---:|---:|---:|---:|---:|']
    for p,v in summary.items():lines.append(f"|{p}|{100*v['f1']:.2f}|{v['generated_tokens']:.2f}|{v['prompt_tokens']:.2f}|{v['tool_calls']:.2f}|{v['zero_tool_episodes']}|{v['errors']}|")
    lines += ['',result['note'],'','Goal remains active; this report does not establish independent generalization.']
    (out/'campaign_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'run':args.run,'policies':summary,'goal_achieved':False},indent=2))
if __name__=='__main__':main()
