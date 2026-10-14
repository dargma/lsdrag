"""Audit successful teacher labels; never imports held-out question answers."""
import argparse
import json
from pathlib import Path
from lsdrag.decision import action_label, label_gate
from lsdrag.io import read_jsonl, dump_json
from lsdrag.metrics import official_scorer, answer_scores


def audit(run, gold, scorer, allowed=None):
    records, scored = [], []
    for q in read_jsonl(gold):
        if allowed is not None and q['qid'] not in allowed:
            continue
        path = Path(run) / (q['qid'] + '.json')
        if not path.exists():
            continue
        pred = json.loads(path.read_text())
        if pred['status'] == 'failed':
            raise ValueError(f'Unresolved infrastructure failure: {path}')
        scores = answer_scores(pred, q, scorer)
        scored.append({'qid':q['qid'], **scores})
        if scores['em'] == 1:
            records.append({'qid':q['qid'], 'source':str(path), 'actions':[a for i,call in enumerate(pred['reader_calls']) if not (pred.get('stop_reason') and i==len(pred['reader_calls'])-1) and (a := action_label(call['message'])) is not None], 'answerable':q.get('answerable',True)})
    return records, scored


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='append',required=True)
    p.add_argument('--gold',action='append',required=True)
    p.add_argument('--allowed-split',type=Path)
    p.add_argument('--upstream',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if len(a.run)!=len(a.gold): raise ValueError('Pair each run with gold')
    allowed=None
    if a.allowed_split:
        allowed=set(json.loads(a.allowed_split.read_text())['splits']['train'])
    scorer=official_scorer(a.upstream)
    records,scored=[],[]
    for run,gold in zip(a.run,a.gold):
        r,s=audit(run,gold,scorer,allowed);records+=r;scored+=s
    report={'completed':len(scored),'em':sum(s['em'] for s in scored)/len(scored) if scored else None,'f1':sum(s['f1'] for s in scored)/len(scored) if scored else None,'successful_trajectories':len(records),'gate':label_gate(records),'successes':records,'question_metrics':scored}
    dump_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'successes','question_metrics'}}))

if __name__=='__main__': main()
