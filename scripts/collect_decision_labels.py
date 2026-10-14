"""Bounded teacher collection: frozen M3 -> MMQA -> one MultiHop fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from audit_decision_labels import audit
from lsdrag.decision import label_gate
from lsdrag.io import read_jsonl,write_jsonl,dump_json,sha256
from lsdrag.metrics import official_scorer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-root',type=Path,required=True)
    a=p.parse_args();r=a.work_root;repo=Path(__file__).resolve().parents[1]
    python=r/'embedding-env/bin/python';output=r/'results/jev_v1.0'
    plan=json.loads((repo/'lab/registry/jev_v1.0/label_collection_plan.json').read_text())
    frozen=repo/'lab/registry/jev_v1.0/split.json';allowed=set(json.loads(frozen.read_text())['splits']['train'])
    scorer=official_scorer(r/'upstream/m3docrag')
    sources=[(r/'results/three_arm_search_linked100/m3docvqa'/arm,r/'data/manifests/search_linked100/m3docvqa/gold.jsonl',allowed) for arm in ['B0','C1','G1']]
    def pooled():
        records=[];metrics=[]
        for run,gold,subset in sources:
            rows,scores=audit(run,gold,scorer,subset);records+=rows;metrics+=scores
        report={'completed_trajectories':len(metrics),'successes':records,'successful_trajectories':len(records),'gate':label_gate(records),'sources':[{'run':str(run),'gold_sha256':sha256(gold),'allowed_old_train_only':subset is not None} for run,gold,subset in sources],'null_scoring':'Exact original answer Insufficient information. under official normalization; empty list is incorrect; null successes excluded from answerable/action gate'}
        dump_json(output/'pooled_label_audit.json',report)
        print(json.dumps({'stage':'audit','completed':len(metrics),'gate':report['gate']}),flush=True)
        return report
    def solve(name,questions,corpus,indexes,arm,benchmark='m3docvqa'):
        run=output/name
        expected=read_jsonl(questions)
        existing=[run/(q['qid']+'.json') for q in expected]
        if all(f.exists() for f in existing):
            if any(json.loads(f.read_text())['status']=='failed' for f in existing): raise RuntimeError('Existing infrastructure failure; manual inspection required')
            return run
        if run.exists(): raise RuntimeError(f'Partial run must be inspected, never silently retried: {run}')
        subprocess.run([str(python),'scripts/run_experiment.py','--arm',arm,'--benchmark',benchmark,'--corpus',str(corpus),'--indexes',str(indexes),'--questions',str(questions),'--output',str(run),'--cache-dir',str(r/'hf')],cwd=repo,check=True)
        return run
    # The already launched 82-question run must finish before any new teacher run.
    run=output/'teacher_m3_train82';qs=r/'data/manifests/jev_v1.0/fresh_m3_train82'
    for _ in range(180):
        paths=[run/(q['qid']+'.json') for q in read_jsonl(qs/'questions.jsonl')]
        if any(f.exists() and json.loads(f.read_text())['status']=='failed' for f in paths): raise RuntimeError('M3 infrastructure failure')
        if all(f.exists() for f in paths): break
        time.sleep(10)
    else: raise RuntimeError('M3 collection did not finish within bounded wait')
    sources.append((run,qs/'gold.jsonl',None));report=pooled()
    if not report['gate']['passed']:
        qs=r/'data/manifests/jev_v1.0/mmqa_train_qc'
        run=solve('teacher_mmqa_train37',qs/'questions.jsonl',r/'data/ir_final288',r/'indexes_final288','G1')
        sources.append((run,qs/'gold.jsonl',None));report=pooled()
    if not report['gate']['passed']:
        # The existing M3 corpus/index is unchanged. This is a new text-only
        # training corpus, not a re-index of the M3 evaluation documents.
        corpus=r/'data/jev_multihop_corpus';index=r/'indexes_jev_multihop/gemma_text'
        if not (index/'manifest.json').exists():
            subprocess.run([str(python),'scripts/build_index.py','--corpus',str(corpus),'--output',str(index),'--channel','gemma','--modality','text','--cache-dir',str(r/'hf'),'--batch-size','32'],cwd=repo,check=True)
        qs=r/'data/manifests/jev_v1.0/multihop_train1000';questions=read_jsonl(qs/'questions.jsonl');gold=read_jsonl(qs/'gold.jsonl')
        for start in range(0,1000,100):
            block=qs/f'block_{start//100+1:02d}'
            write_jsonl(block/'questions.jsonl',questions[start:start+100]);write_jsonl(block/'gold.jsonl',gold[start:start+100])
            run=solve(f'teacher_multihop_{start//100+1:02d}',block/'questions.jsonl',corpus,index.parent,'C1','multihoprag')
            sources.append((run,block/'gold.jsonl',None));report=pooled()
            if report['gate']['passed']: break
    dump_json(output/'collection_complete.json',{'gate':report['gate'],'pooled_audit_sha256':sha256(output/'pooled_label_audit.json'),'next':'train fixed head once' if report['gate']['passed'] else 'NoGo: insufficient labels; do not run full head comparison'})

if __name__=='__main__': main()
