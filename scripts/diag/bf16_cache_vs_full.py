import json,sys,torch
from transformers import AutoModelForCausalLM
spec=json.load(open(sys.argv[1]))
path=[p for p in spec['group_files'] if '3hop2__713535' in p][0]
g=torch.load(path,weights_only=True,map_location='cpu')[0]; step=g['steps'][2]; seg=step['segment']; ctx,tok=seg['context_ids'],seg['tokens']
print('ctx',len(ctx),'tok',len(tok),'old@76',seg['old_logp'][76],'old 70..80',[round(x,3) for x in seg['old_logp'][70:80]])
for dt in (torch.bfloat16,torch.float32):
    m=AutoModelForCausalLM.from_pretrained(spec['model'],torch_dtype=dt).cuda().eval()
    with torch.no_grad():
        full=m(input_ids=torch.tensor([ctx+tok],device='cuda')).logits[0].float().log_softmax(-1)
        f=[float(full[len(ctx)-1+i,tok[i]]) for i in range(len(tok))]
        out=m(input_ids=torch.tensor([ctx],device='cuda'),use_cache=True); past=out.past_key_values; lg=out.logits[0,-1].float()
        inc=[]
        for i,t in enumerate(tok):
            inc.append(float(lg.log_softmax(-1)[t]))
            out=m(input_ids=torch.tensor([[t]],device='cuda'),past_key_values=past,use_cache=True); past=out.past_key_values; lg=out.logits[0,-1].float()
    d=[abs(a-b) for a,b in zip(f,inc)]
    print(str(dt),'full@76',round(f[76],3),'incremental@76',round(inc[76],3),'| max |full-inc| over segment',round(max(d),3),'at',d.index(max(d)),'mean',round(sum(d)/len(d),4))
    print('   70..80 full',[round(x,2) for x in f[70:80]]); print('   70..80 inc ',[round(x,2) for x in inc[70:80]])
    del m; torch.cuda.empty_cache()
