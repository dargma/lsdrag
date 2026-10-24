"""Verify train48 native episodes and causal read-feature provenance before fit."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from lsdrag.text_decision_benchmark import canonical

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    run=ROOT/'results/decision_h2_read48_20261024'
    spec=json.loads((run/'spec.json').read_text());status=json.loads((run/'status.json').read_text())
    assert status['phase']=='complete' and status['episodes']==48
    assert spec==json.loads((REPO/'lab/registry/decision_h2_read48_20261024/spec.json').read_text())
    for p,h in spec['source_sha256'].items():assert sha(REPO/p)==h,p
    for p in spec['policies'].values():assert sha(p['checkpoint'])==p['sha256']
    sys.path.insert(0,str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact,compute_f1
    raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open())}
    groups={k:dict(n=0,total_generated=0) for k in ['correct','incorrect','partial']}
    split={k:dict(read_qids=set(),calls=0,tokens=0,representable=0) for k in ['fit','diagnostic']}
    hashes={};features_seen=set();f1total=0;errors=0
    for qid in spec['qids']:
        q=raw[qid];path=run/'episodes'/f'h2_{qid}.json';hashes[str(path)]=sha(path);e=json.loads(path.read_text())
        answers=[q['answer']]+q['answer_aliases'];f1=max(compute_f1(a,e['answer']) for a in answers);em=max(compute_exact(a,e['answer']) for a in answers)
        assert abs(f1-e['f1'])<1e-10;f1total+=f1;errors+=bool(e['errors'])
        for label in ['correct' if em else 'incorrect']+(['partial'] if not em and 0<f1<1 else []):groups[label]['n']+=1;groups[label]['total_generated']+=e['generated_tokens']
        calls=[c for c in e['reader_calls'] if 'raw_json' in c];assert len(calls)==len(e['decisions']);assert sum(c['generated_tokens'] for c in calls)==e['generated_tokens']
        messages=[{'role':'system','content':(REPO/'prompts/text_rag_integrated_v5.txt').read_text()},{'role':'user','content':q['question']}]
        history=[];bucket=split['fit' if qid in spec['fit_qids'] else 'diagnostic']
        for d,c in zip(e['decisions'],calls):
            if d['action']==3:continue
            turn=len(history);step=e['trajectory'][turn]
            if step['tool_name']=='read_chunk':
                fp=run/'read_features'/f'{qid}_{turn}.pt';hashes[str(fp)]=sha(fp);features_seen.add(fp.name)
                rec=torch.load(fp,map_location='cpu',weights_only=True)
                assert rec['prefix_sha256']==hashlib.sha256(canonical(messages).encode()).hexdigest()
                assert rec['arguments']==step['arguments'] and rec['error'] is None and rec['generated_tokens']==c['generated_tokens']
                candidates=list(dict.fromkeys(row['chunk_id'] for prior in reversed(history) for row in prior['result'] if isinstance(row,dict) and isinstance(row.get('chunk_id'),str)))
                assert candidates==[x['chunk_id'] for x in rec['candidates']]
                assert rec['candidate_features'].shape==(len(candidates),2560) and rec['pack'].shape==(37,2560)
                assert torch.isfinite(rec['candidate_features']).all() and torch.isfinite(rec['pack']).all()
                bucket['read_qids'].add(qid);bucket['calls']+=1;bucket['tokens']+=c['generated_tokens'];bucket['representable']+=all(x in candidates for x in rec['arguments']['chunk_ids'])
            history.append(step)
            tc={'id':f'call_{len(history)-1}','type':'function','function':{'name':step['tool_name'],'arguments':canonical(step['arguments'])}}
            messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]},{'role':'tool','tool_call_id':tc['id'],'content':canonical(step['result'])}])
    assert features_seen=={p.name for p in (run/'read_features').glob('*.pt')}
    for g in groups.values():g['mean_generated']=g['total_generated']/g['n'] if g['n'] else None
    for b in split.values():b['read_questions']=len(b.pop('read_qids'))
    gate=split['fit']['read_questions']>=12 and split['diagnostic']['read_questions']>=6 and errors==0
    report=dict(f1=f1total/48,mean_generated=status['generated']/48,errors=errors,correctness=groups,coverage=split,fit_coverage_gate=gate,
                interpretation='H2 reference collection only. Coverage is not learned pointer quality or token savings. Diagnostic questions are training-source probes, not independent test.',artifact_sha256=hashes)
    out=REPO/'lab/reports/decision_h2_read48_20261024';out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2))

if __name__=='__main__':main()
