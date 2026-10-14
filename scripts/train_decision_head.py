"""Extract frozen Gemma features from successful train trajectories, then CE head."""
import argparse
import json
import random
from pathlib import Path
import torch
from lsdrag.agent import AgentConfig,tool_schemas
from lsdrag.decision import ACTIONS,trajectory_states
from lsdrag.hf_reader import HFReader
from lsdrag.io import read_jsonl,dump_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.work_root
    audit=json.loads(a.audit.read_text())
    if not audit['gate']['passed']: raise ValueError('Label gate failed: no head training')
    if a.output.exists(): raise ValueError('Refuse to overwrite/retrain head')
    random.seed(20261014);torch.manual_seed(20261014);torch.cuda.manual_seed_all(20261014)
    repo=Path(__file__).resolve().parents[1];prompt=(repo/'prompts/search_linked.txt').read_text()
    plan=json.loads((repo/'lab/registry/jev_v1.0/label_collection_plan.json').read_text())
    splits=json.loads((repo/'lab/registry/jev_v1.0/split.json').read_text())['splits']
    forbidden=set(plan['fresh_m3_test_qids']) | set(splits['test']) | set(splits['dev'])
    if any(s['qid'] in forbidden for s in audit['successes']):
        raise ValueError('Held-out/development question in training audit')
    figures={f['figure_id']:f for f in read_jsonl(r/'data/ir_final288/figures.jsonl')}
    reader=HFReader(str(r/'hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2'),AgentConfig())
    features=[];labels=[];rows=[]
    for success in audit['successes']:
        path=Path(success['source']);pred=json.loads(path.read_text())
        for state in trajectory_states(pred,prompt,figures):
            if state['forced_final'] or state['action'] is None: continue
            inputs,output,feature,seconds=reader.prefill(state['messages'],tool_schemas('G1' if pred['arm']=='G1' else pred['arm'],'search-linked'))
            features.append(feature[0].cpu().float());labels.append(ACTIONS.index(state['action']))
            rows.append({'qid':state['qid'],'loop':state['loop'],'action':state['action'],'source':str(path),'source_sha256':sha256(path),'prefill_seconds':seconds})
            del inputs,output,feature
        print(json.dumps({'feature_trajectories':len({x['source'] for x in rows}),'feature_turns':len(rows)}),flush=True)
    x=torch.stack(features);y=torch.tensor(labels)
    # A single fixed fit; no held-out result, calibration sweep or repeated fit.
    head=torch.nn.Linear(x.shape[1],len(ACTIONS))
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
    for epoch in range(100):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),y);loss.backward();optimizer.step()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'weight':head.weight.detach(),'bias':head.bias.detach(),'actions':list(ACTIONS)},a.output)
    torch.save({'features':x,'labels':y},a.output.with_suffix('.features.pt'))
    dump_json(a.output.with_suffix('.json'),{'seed':20261014,'backbone_frozen':True,'fit':'one full-batch AdamW run, 100 epochs, lr .001, weight decay .01, plain CE; no test tuning','training_loss':float(loss.detach()),'turns':len(rows),'audit_sha256':sha256(a.audit),'features':rows,'gate':audit['gate'],'checkpoint_sha256':sha256(a.output),'probability_meaning':'uncalibrated teacher-action relative probability, not answer correctness'})

if __name__=='__main__': main()
