"""Same-data offline architecture comparison from selected pre-GRPO head."""
import copy,json,os,time
from pathlib import Path
import torch
from lsdrag.decision_attention_adapter import AttentionAdapter
from lsdrag.decision_state_learning import action_values,expected_utility_loss
from lsdrag.decision_policy import distribution
from lsdrag.decision_policy_followup import state_weights
from evaluate_decision_value_heads import load_head
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_attention_20261024'
def main():
    p=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for f,h in p['source_sha256'].items():assert sha(REPO/f)==h,f
    for f,h in p['input_sha256'].items():assert sha(Path(f))==h,f
    out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p);start=time.monotonic();torch.set_num_threads(2)
    def phase(n,**kw):save(out/'status.json',dict(phase=n,pid=None if n in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw))
    phase('loading')
    try:
        rs=torch.load(p['outcomes'],map_location='cpu',weights_only=True);assert {r['qid'] for r in rs}==set(p['train']);assert not set(p['train'])&set(p['validation']+p['test'])
        x=torch.stack([r['pack'] for r in rs]).float();mask=torch.stack([r['legal'] for r in rs]);w=state_weights([r['qid'] for r in rs],'cpu');q,_,_=action_values(rs,.03)
        base=load_head(p['reference_spec']);ref=distribution(base(x,mask)[0],mask).probs.detach();candidates={};audits={}
        for variant in ['question_attention','action_attention','action_set']:
            phase('training',candidate=variant);torch.manual_seed(p['seed'])
            h=AttentionAdapter(base,variant)
            with torch.no_grad():assert torch.equal(h(x,mask)[0],base(x,mask)[0])
            buffers={k:v.clone() for k,v in h.named_buffers()};opt=torch.optim.AdamW(h.parameters(),lr=1e-4);curve=[]
            for epoch in range(100):
                opt.zero_grad();loss=expected_utility_loss(h(x,mask)[0],mask,q,ref,w);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1);opt.step()
                if epoch%20==0 or epoch==99:curve.append(dict(epoch=epoch,loss=float(loss.detach())))
            assert all(torch.equal(v,buffers[k]) for k,v in h.named_buffers());assert all(torch.isfinite(v).all() for v in h.state_dict().values())
            with torch.no_grad():
                changed=x.clone();changed[:,5:37]=changed[:,5:37].roll(1,0)
                sensitivity=float((h(x,mask)[0]-h(changed,mask)[0])[mask].abs().mean())
                assert sensitivity>0,'Attention path unused'
            cp=out/(variant+'.pt');torch.save(h.state_dict(),cp)
            spec=dict(kind='attention_adapter',variant=variant)
            candidates[variant]=dict(**spec,checkpoint=str(cp),sha256=sha(cp));audits[variant]=dict(initial_logits_exact=True,normalization_fixed=True,parameters=sum(v.numel() for v in h.parameters()),context_sensitivity=sensitivity,curve=curve)
        save(out/'manifest.json',dict(candidates=candidates,audits=audits,plan=p,new_llm_rollouts=0));phase('complete')
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
