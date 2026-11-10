import json,sys,collections
from types import SimpleNamespace
import torch, numpy as np, torch.nn.functional as F
from lsdrag.fullft import SampledReader, replay_v22_step
spec=json.load(open(sys.argv[1])); arm=spec['arm']
reader=SampledReader(spec['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384)); model=reader.model
W=model.lm_head.weight.float()
gaps=collections.defaultdict(list); q=[]; scale=[]
with torch.no_grad():
    for path in spec['group_files']:
        for g in torch.load(path,weights_only=True,map_location='cpu'):
            for step in g['steps']:
                if arm=='native':
                    ev,_=replay_v22_step(model,step)
                    for e in ev: gaps[e['kind']].append(float(e['new'])-float(e['old']))
                seg=step['segment']; ctx,tok=seg['context_ids'],seg['tokens']
                want=[i for i,k in enumerate(seg['kinds']) if k=='free']
                if not want: continue
                h=model.model(input_ids=torch.tensor([ctx+tok],device='cuda'),use_cache=False,return_dict=True).last_hidden_state[0]
                pos=torch.tensor([len(ctx)-1+i for i in want],device='cuda'); t=torch.tensor([tok[i] for i in want],device='cuda')
                l16=model.lm_head(h[pos]).float(); l32=F.linear(h[pos].float(),W)
                a=l16.log_softmax(-1).gather(1,t[:,None])[:,0]; b=l32.log_softmax(-1).gather(1,t[:,None])[:,0]
                q+= (a-b).abs().tolist(); scale.append(float(l16.max()))
                # logit grid: distinct spacing of bf16 logits near the max
if gaps:
    for k,v in gaps.items():
        a=np.abs(v); print(arm,k,'n',len(v),'mean',a.mean().round(4),'p50',np.median(a).round(4),'p99',np.percentile(a,99).round(4),'max',a.max().round(4),'>0.05',int((a>.05).sum()),'>0.182',int((a>.182).sum()))
q=np.array(q); print('bf16-head vs fp32-head logp on identical hidden: n',len(q),'mean',q.mean().round(4),'p50',np.median(q).round(4),'p99',np.percentile(q,99).round(4),'max',q.max().round(4),'>0.05',int((q>.05).sum()),'typical max logit',np.median(scale).round(1))
