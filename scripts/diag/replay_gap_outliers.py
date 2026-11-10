import json,sys
from types import SimpleNamespace
import torch
from lsdrag.fullft import SampledReader
spec=json.load(open(sys.argv[1]))
reader=SampledReader(spec['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384)); model=reader.model; tk=reader.tokenizer
with torch.no_grad():
    for path in spec['group_files']:
        for g in torch.load(path,weights_only=True,map_location='cpu'):
            for si,step in enumerate(g['steps']):
                seg=step['segment']; ctx,tok=seg['context_ids'],seg['tokens']
                want=[i for i,k in enumerate(seg['kinds']) if k=='free']
                if not want: continue
                h=model.model(input_ids=torch.tensor([ctx+tok],device='cuda'),use_cache=False,return_dict=True).last_hidden_state[0]
                lp=model.lm_head(h[[len(ctx)-1+i for i in want]]).float().log_softmax(-1)
                for r,i in enumerate(want):
                    d=float(lp[r,tok[i]])-seg['old_logp'][i]
                    if abs(d)>1:
                        top=lp[r].topk(3)
                        print(g['episode']['qid'],'rep',g['episode']['rep'],'step',si,'i',i,'of',len(tok),'kinds',seg['kinds'][max(0,i-2):i+3],'tok',repr(tk.decode([tok[i]])),'old',round(seg['old_logp'][i],3),'new',round(float(lp[r,tok[i]]),3),'top',[(repr(tk.decode([int(t)])),round(float(v),3)) for v,t in zip(top.values,top.indices)],'| text',repr(tk.decode(tok[max(0,i-8):i+3])),'errors',g['episode']['errors'])
