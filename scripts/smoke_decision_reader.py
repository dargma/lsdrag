"""Two train-only smoke states, not a benchmark or a trained head evaluation."""
import argparse
import json
from pathlib import Path
import torch
from lsdrag.agent import AgentConfig, tool_schemas
from lsdrag.decision import ACTIONS,trajectory_states
from lsdrag.hf_reader import HFReader
from lsdrag.io import read_jsonl,dump_json

p=argparse.ArgumentParser()
p.add_argument('--work-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); r=a.work_root
repo=Path(__file__).resolve().parents[1]
allowed=set(json.loads((repo/'lab/registry/jev_v1.0/split.json').read_text())['splits']['train'])
figures={f['figure_id']:f for f in read_jsonl(r/'data/ir_final288/figures.jsonl')}
prompt=(repo/'prompts/search_linked.txt').read_text()
states={}
for path in sorted((r/'results/three_arm_search_linked100/m3docvqa/G1').glob('*.json')):
    pred=json.loads(path.read_text())
    if pred.get('qid') not in allowed: continue
    for state in trajectory_states(pred,prompt,figures):
        if state['forced_final']: continue
        image=any(isinstance(m.get('content'),list) for m in state['messages'])
        states.setdefault('image' if image else 'text',state)
    if len(states)==2: break
assert len(states)==2
reader=HFReader(str(r/'hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2'),AgentConfig(max_output_tokens=128))
rows=[]
for kind,state in states.items():
    choice,usage=reader.chat(state['messages'],tool_schemas('G1','search-linked'))
    rows.append({'kind':kind,'mode':'baseline','qid':state['qid'],'loop':state['loop'],'choice':choice,'usage':usage})
    # Deterministic synthetic head tests control flow only, not learned quality.
    head=torch.nn.Linear(reader.feature.shape[-1],len(ACTIONS),device='cuda',dtype=torch.float32)
    with torch.no_grad():
        head.weight.zero_();head.bias.zero_();head.bias[ACTIONS.index('semantic_search' if kind=='image' else 'FINAL')]=10
    reader.head=head.eval()
    choice,usage=reader.chat(state['messages'],tool_schemas('G1','search-linked'))
    rows.append({'kind':kind,'mode':'synthetic_control_flow_head','qid':state['qid'],'loop':state['loop'],'choice':choice,'usage':usage})
    reader.head=None
    dump_json(a.output,{'smoke_only':True,'trained_head':False,'rows':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    print(json.dumps({'kind':kind,'passed':True}),flush=True)
