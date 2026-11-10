import json,sys,collections
from pathlib import Path
from types import SimpleNamespace
import torch, numpy as np
from lsdrag.fullft import SampledReader, replay_v22_step
from lsdrag.chunk_pointer import ChunkPointer
from lsdrag.decision_policy import PolicyHead
spec=json.load(open(sys.argv[1])); arm=spec['arm']
reader=SampledReader(spec['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384)); model=reader.model
head=pointer=None
if arm=='head':
    ck=torch.load(spec['head_checkpoint'],weights_only=True,map_location='cpu')
    head=PolicyHead();head.load_state_dict(ck['tool']);head=head.cuda().eval()
    pointer=ChunkPointer();pointer.load_state_dict(ck['pointer']);pointer=pointer.cuda().eval()
gaps=collections.defaultdict(list); worst=[]
with torch.no_grad():
    for path in spec['group_files'][:int(sys.argv[2])]:
        for g in torch.load(path,weights_only=True,map_location='cpu'):
            for si,step in enumerate(g['steps']):
                ev,_=replay_v22_step(model,step,head,pointer)
                seg=step['segment']
                for e in ev:
                    d=float(e['new'])-float(e['old']); gaps[e['kind']].append(d)
                    if abs(d)>.05: worst.append((round(d,3),e['kind'],g['episode']['qid'],g['episode']['rep'],si,round(float(e['old']),3),len(seg['context_ids']),len(seg['tokens'])))
for k,v in gaps.items():
    a=np.abs(v); print(k,'n',len(v),'mean',a.mean().round(4),'p50',np.median(a).round(4),'p99',np.percentile(a,99).round(4),'max',a.max().round(4),'>0.05',int((a>.05).sum()),'signed mean',np.mean(v).round(4))
worst.sort(key=lambda x:-abs(x[0]));print(len(worst));[print(w) for w in worst[:25]]
