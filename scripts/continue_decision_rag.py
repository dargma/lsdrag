"""Sequential stage2 executor after a verified stage1 checkpoint; no GPU overlap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from checkpoint_decision_rag import checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--common-args',type=Path,required=True);a=p.parse_args()
    common=json.loads(a.common_args.read_text());value=lambda name:Path(common[common.index('--'+name)+1])
    run=value('output');repo=Path(__file__).resolve().parents[1];report=repo/'lab/reports/decision_rag_20261015'
    diagnosis=json.loads((report/'stage1/diagnostic.json').read_text())
    if not diagnosis['stage1_all20_complete']:raise ValueError('Stage1 paired20 must be complete')
    if not (report/'checkpoint_stage1_complete.json').exists():raise ValueError('Stage1 checkpoint required')
    # Caller verifies no live collector before launch; an exclusive lock also rejects duplicate coordinators.
    import fcntl
    lock=(run/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=run/'coordinator_status.json'
    def status(phase,**extra):
        row={'phase':phase,'updated_epoch':time.time(),'pid':os.getpid(),**extra}
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(state);print(json.dumps(row),flush=True)
    def execute(phase,start=0,limit=20):
        log=run.parent.parent/'logs'/f'{phase}_{start:03d}_{start+limit:03d}.log'
        cmd=[sys.executable,'-u','scripts/run_decision_rag_stages.py',*common,'--phase',phase,'--start',str(start),'--limit',str(limit)]
        with log.open('w') as stream:
            process=subprocess.Popen(cmd,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src'})
            status('running_'+phase,child_pid=process.pid,log=str(log),command=cmd)
            code=process.wait()
        if code:raise RuntimeError(f'{phase} failed exit{code}; inspect {log}')
    import torch
    from lsdrag.head_experiment import validate_teacher_artifact,validate_completed_prefix
    manifest=json.loads((run/'manifest.json').read_text());contract=manifest['feature_contract'];order=manifest['train_question_order']
    def coverage(expected):
        seen={name:set() for name in contract['actions']};qids=[];allq=set()
        for qid in order:
            pt=run/'teacher'/(qid+'.pt');js=pt.with_suffix('.json')
            if not pt.exists() and not js.exists():continue
            if not pt.exists() or not js.exists():raise ValueError('Unpaired teacher artifact')
            artifact=torch.load(pt,map_location='cpu',weights_only=True);validate_teacher_artifact(artifact,qid,contract)
            episode=json.loads(js.read_text())
            if episode['qid']!=qid:raise ValueError('Teacher episode identity mismatch')
            qids.append(qid)
            for r in artifact['records']:
                if r.get('forced_final') or r['selector']!='generated':raise ValueError('Invalid teacher label')
                seen[r['action']].add(qid);allq.add(qid)
        if len(qids)!=expected or qids!=order[:expected]:raise ValueError('Incorrect teacher prefix')
        if expected>=60:validate_completed_prefix(qids,order)
        return {'completed_teacher_questions':expected,'distinct_teacher_qids':len(allq),
                'distinct_qids_per_action':{k:len(v) for k,v in seen.items()},
                'passed':len(allq)>=30 and all(len(v)>=5 for v in seen.values())}
    def report_full():
        subprocess.run([sys.executable,'scripts/report_full_head_pilot.py','--run',str(run),'--data',str(value('train-questions').parent),'--report',str(report/'head')],check=True,cwd=repo)
    try:
        coverage(20)
        for end in [60,80,100]:
            start=20 if end==60 else end-20
            execute('teacher',start,end-start);gate=coverage(end);report_full()
            checkpoint(run,'teacher_'+str(end),{'coverage':gate,'next':'train' if gate['passed'] else ('teacher_'+str(end+20) if end<100 else 'coverage_no_go')})
            if gate['passed']:break
        if not gate['passed']:
            status('coverage_no_go',coverage=gate)
            checkpoint(run,'coverage_no_go',{'coverage':gate,'training_completed':False,'evaluation_completed':False})
            return
        execute('train');training=json.loads((run/'training.json').read_text())
        if not training['gate']['passed'] or len(training['runs'])!=3:raise ValueError('Training did not produce all3 heads')
        report_full();checkpoint(run,'trained_3seeds',{'coverage':gate,'seeds':[r['seed'] for r in training['runs']],'next':'paired_dev20'})
        execute('evaluate');evaluation=json.loads((run/'evaluation.json').read_text())
        if len(evaluation['policies'])!=6 or any(r['n']!=20 for r in evaluation['policies'].values()):raise ValueError('Incomplete6-policy paired dev20')
        report_full();status('stage2_evaluated',policies=evaluation['policies'])
        checkpoint(run,'stage2_evaluated',{'evaluation':evaluation,'next':'manual_completion_audit_and_independent_argument_ablation'})
    except BaseException as error:
        status('execution_failed',error=type(error).__name__+': '+str(error));raise

if __name__=='__main__':main()
