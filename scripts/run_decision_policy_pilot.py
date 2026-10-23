"""Frozen, bounded v1.2 experiment: readout diagnostics -> outcome CE -> on-policy PPO."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from lsdrag.decision_policy import (PolicyReader, PolicyHead, distribution, rollout,
                                    returns_for_episode, clipped_policy_loss)
from lsdrag.text_integrated import ACTIONS, options_for, run_episode
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/text_integrated_20261016_v5_fast_only'
ORIGIN=ROOT/'results/improvement_v3_merged_20261017'
REG=REPO/'lab/registry/decision_policy_20261023'
SEED=20261023


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def prepare():
    split=json.loads((REPO/'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
    exposed=set()
    for folder in [REPO/'lab',ROOT/'results']:
        for p in folder.rglob('*.json'):
            exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
    train=[json.loads(s) for s in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').read_text().splitlines()]
    dev=[json.loads(s) for s in (ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').read_text().splitlines()]
    families={d['id'] for q in train+dev if q['id'] in exposed for d in q['question_decomposition']}
    eligible=[q for q in dev if q['answerable'] and len(q['question_decomposition'])==2 and q['id'] not in exposed and not {d['id'] for d in q['question_decomposition']}&families]
    test=[];used=set()
    for q in sorted(eligible,key=lambda q:hashlib.sha256(('decision-policy-v1.2:'+q['id']).encode()).hexdigest()):
        fs={d['id'] for d in q['question_decomposition']}
        if not fs&used:test.append(q['id']);used.update(fs)
        if len(test)==20:break
    assert len(test)==20
    m={'profile':'decision_policy_20261023/pilot1','seed':SEED,'train':split['train'],'validation':split['validation'],'test':test,
       'excluded_qids':sorted(exposed),'eligible_fresh2hop':len(eligible), 'test_frozen_before_outcomes':True,
       'model':json.loads((PARENT/'manifest.json').read_text())['model'], 'head_sha256':sha(ORIGIN/'head.pt'),
       'teacher_pairs':32,'branch_states_per_question':2,'branch_repeats':2,'branch_cap':384,
       'rl_iterations':3,'episodes_per_iteration':48,'max_tool_calls':8,'max_output_tokens':512,
       'context_limit':16384,'worker_seconds':3600,'generated_token_cap':600000,'input_token_cap':12000000,
       'cost_weight':.005,'cost_units':'tool_call + decoded_tokens/100 + prefill_and_prefix_tokens/10000',
       'quality':'answer_F1','success_label':'answer_EM','discount':1.,'ppo_lr':.0001,'ppo_epochs':4,'ppo_clip':.2,
       'entropy_coef':.01,'bc_KL_coef':.02,'exploration_epsilon':.05,'value_coef':.5,
       'representation_epochs':100,'representation_lr':.001,'architecture_selection':'validation fixed-state macroF1 among last_mlp and attention32; tie last_mlp',
       'outcome_ce':'per-state softmax(mean branch return / .1); same base architecture; 100 epochs lr1e-4',
       'test_policies':['native','native_fast','historical_bc','bc','outcome_ce','ppo'],
       'probability_semantics':'policy probabilities are not success probabilities; auxiliary success is uncalibrated frozen-BC-continuation estimate',
       'source_sha256':{n:sha(REPO/n) for n in ['src/lsdrag/decision_policy.py','scripts/run_decision_policy_pilot.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']}}
    REG.mkdir(parents=True,exist_ok=True)
    target=REG/'manifest.json'
    if target.exists():raise FileExistsError('Do not overwrite frozen manifest')
    save(target,m);print(target)


def macro(y,p):
    cm=np.zeros((4,4),int)
    for a,b in zip(y,p):cm[a,b]+=1
    return {'n':len(y),'macro_f1':float(np.mean([2*cm[i,i]/max(1,cm[i].sum()+cm[:,i].sum()) for i in range(4)])),
            'confusion':cm.tolist(),'premature_final':int(sum(a!=3 and b==3 for a,b in zip(y,p)))}


def train_bc(x,y,legal,arch,feature,mean,scale):
    torch.manual_seed(SEED);head=PolicyHead(x.shape[-1],arch,feature)
    head.mean.copy_(mean);head.scale.copy_(scale)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    for _ in range(100):
        opt.zero_grad();logits,*_=head(x,legal);loss=F.cross_entropy(logits,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    return head


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.prepare:prepare();return
    m=json.loads((REG/'manifest.json').read_text())
    for n,h in m['source_sha256'].items():assert sha(REPO/n)==h,n
    assert sha(ORIGIN/'head.pt')==m['head_sha256']
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    import fcntl
    lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    save(out/'manifest.json',m)
    start=time.perf_counter();deadline=None;counts={'episodes':0,'generated':0,'inputs':0};embedder=None
    def phase(name,**kw):
        state={'phase':name,'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-start,**counts,**kw}
        save(out/'status.json',state);print(json.dumps(state),flush=True)
    def reserve():
        if time.perf_counter()>=deadline:raise TimeoutError('3600s worker cap')
        if counts['generated']+9*512>m['generated_token_cap'] or counts['inputs']+9*16448>m['input_token_cap']:raise TimeoutError('token cap reserve')
    def charge(ep):
        counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens']
        assert counts['generated']<=m['generated_token_cap'] and counts['inputs']<=m['input_token_cap']
    sys.path.insert(0,str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact,compute_f1
    raw={}
    for name in ['train','dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q=json.loads(line)
            if q['id'] in set(m['train']+m['validation']+m['test']):raw[q['id']]=q
    def score(ep,q):
        aliases=[q['answer']]+q['answer_aliases'];gold={f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']};pred=set(ep['support_ids'])
        ep.update(qid=q['id'],em=max(compute_exact(x,ep['answer']) for x in aliases),f1=max(compute_f1(x,ep['answer']) for x in aliases),support_f1=2*len(pred&gold)/max(1,len(pred)+len(gold)))
        return ep
    try:
        phase('loading')
        reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384))
        embedder=V11ProcessEmbedder();save(out/'embedding_runtime.json',embedder.ready)
        prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text()
        hist=torch.load(ORIGIN/'head.pt',map_location='cpu',weights_only=True)
        reader.checkpoint={k:hist[k].to('cuda') for k in ['action_weight','action_bias','slot_weight','slot_bias']}
        historical={k:hist[k].to('cuda') for k in ['action_weight','action_bias']}
        deadline=time.perf_counter()+m['worker_seconds']
        cache={}
        def corpus(q):
            if q['id'] not in cache:
                rows=public_paragraphs(q);p=PARENT/'index'/f'{q["id"]}.npy'
                vec=np.load(p) if p.exists() else embedder.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
                cache[q['id']]=(rows,vec)
            return cache[q['id']]
        def episode(qid,head,folder,tag,**kw):
            reserve();q=raw[qid];rows,vec=corpus(q)
            ep,steps,snap=rollout(reader,q['question'],rows,vec,embedder,prompt,head,deadline,**kw)
            score(ep,q);ep['policy']=tag;ep['return']=ep['f1']-m['cost_weight']*ep['cost_units']
            charge(ep);save(out/folder/f'{tag}_{qid}.json',ep)
            return ep,steps,snap
        phase('feature_extraction')
        sets={};parity=[]
        for group in ['train','validation']:
            packs=[];ys=[];masks=[];records=[]
            for qid in m[group]:
                source=json.loads((PARENT/'teacher'/f'{qid}.json').read_text())['records']
                old=np.load(PARENT/'teacher'/f'{qid}.npz')['features']
                for i,r in enumerate(source):
                    reserve();pack,u=reader.capture_pack(r['messages'],r['history'],deadline);counts['inputs']+=u['prompt_tokens'];reader.pending=None
                    packs.append(pack.cpu().clone());ys.append(r['action_index']);masks.append([any(o['action']==x for o in r['options']) for x in ACTIONS]);records.append({'qid':qid,'loop':r['loop']})
                    orig=hist['action_weight']@torch.from_numpy(old[i]).float()+hist['action_bias'];new=hist['action_weight']@pack[0].cpu()+hist['action_bias'];mask=torch.tensor(masks[-1])
                    parity.append({'qid':qid,'loop':r['loop'],'max_abs_error':float((pack[0].cpu()-torch.from_numpy(old[i])).abs().max()),'action_equal':int(orig.masked_fill(~mask,-1e9).argmax())==int(new.masked_fill(~mask,-1e9).argmax())})
                phase('feature_extraction',group=group,qid=qid)
            sets[group]=(torch.stack(packs),torch.tensor(ys),torch.tensor(masks),records)
            torch.save({'pack':sets[group][0].bfloat16(),'labels':sets[group][1],'legal':sets[group][2],'records':records},out/f'{group}_features.pt')
        save(out/'historical_feature_parity.json',parity)
        if not all(x['action_equal'] for x in parity):raise RuntimeError('Historical action parity failed')
        phase('representation_training')
        x,y,legal,_=sets['train'];vx,vy,vl,_=sets['validation'];mean=x[:,0].mean(0);scale=x[:,0].std(0,unbiased=False).clamp_min(.05)
        specs=[('last_linear','linear',0),('question_linear','linear',1),('observation_linear','linear',2),('last16_linear','linear',3),('last_mlp','mlp',0),('observation_mean_mlp','mlp',4),('attention32','attention',0)]
        heads={};diagnostic={}
        for name,arch,idx in specs:
            norm_data=x[:,5:].reshape(-1,x.shape[-1]) if arch=='attention' else x[:,idx]
            mean=norm_data.mean(0);scale=norm_data.std(0,unbiased=False).clamp_min(.05)
            h=train_bc(x,y,legal,arch,idx,mean,scale);heads[name]=h
            with torch.no_grad():pred=h(vx,vl)[0].argmax(-1).tolist()
            diagnostic[name]=macro(vy.tolist(),pred)|{'parameters':sum(p.numel() for p in h.parameters()),'architecture':arch,'feature':idx}
            (out/'heads').mkdir(exist_ok=True)
            torch.save(h.state_dict(),out/f'heads/{name}.pt')
        selected=max(['last_mlp','attention32'],key=lambda n:diagnostic[n]['macro_f1'])
        save(out/'representation_diagnostic.json',{'models':diagnostic,'selected':selected,'diagnostic_only':True})
        bc=heads[selected].to('cuda').eval();save(out/'selected_architecture.json',{'name':selected,'architecture':bc.architecture,'feature':bc.feature,'parameters':sum(p.numel() for p in bc.parameters())})
        # Equal-input native/full-history BC parity on two training episodes.
        phase('functional_episode_smoke')
        for qid in m['train'][:2]:
            q=raw[qid];rows,vec=corpus(q);reserve()
            oldep,_,_=run_episode(q['question'],rows,vec,embedder,reader,prompt,'head','always_generate',deadline)
            oldep=score(oldep,q);oldep['generated_tokens']=oldep['completion_tokens'];charge(oldep);save(out/'smoke'/f'legacy_{qid}.json',oldep)
            newep,_,_=episode(qid,None,'smoke','historical_bc',historical=historical)
            assert oldep['answer']==newep['answer'] and oldep['support_ids']==newep['support_ids']
            assert [(t['tool_name'],t['arguments']) for t in oldep['trajectory']]==[(t['tool_name'],t['arguments']) for t in newep['trajectory']]
        # Native-fast is a new constrained baseline; validate real dispatch.
        smoke,_,_=episode(m['train'][0],None,'smoke','native_fast',native_fast=True)
        if smoke['errors']:raise RuntimeError('native-fast smoke failed')
        phase('branch_collection')
        branch=[]
        for qid in m['train']:
            base,_,snapshots=episode(qid,bc,'train_baseline','bc')
            for si,snap in enumerate(snapshots[:2]):
                opts=options_for(raw[qid]['question'],snap['history']);leg=[any(o['action']==a for o in opts) for a in ACTIONS]
                state=[]
                for ai,ok in enumerate(leg):
                    if not ok:continue
                    for rep in range(2):
                        tag=f's{si}_a{ai}_r{rep}';ep,steps,_=episode(qid,bc,'branches',tag,stochastic=True,initial=snap,first_action=ai)
                        state.append({'action':ai,'return':ep['return'],'em':ep['em'],'cost':ep['cost_units']})
                        if len(branch)*8+len(state)>384:raise RuntimeError('branch cap')
                branch.append({'qid':qid,'state':si,'pack':steps[0]['pack'],'legal':steps[0]['legal'],'outcomes':state})
            phase('branch_collection',qid=qid,states=len(branch))
        torch.save(branch,out/'branch_data.pt')
        # Same collected outcomes provide direct supervised baseline and critic pretraining.
        bx=torch.stack([r['pack'] for r in branch]).to('cuda').float();bl=torch.stack([r['legal'] for r in branch]).to('cuda')
        rewards=torch.full((len(branch),4),-1e9,device='cuda');success=torch.zeros_like(rewards);costs=torch.zeros_like(rewards)
        for i,row in enumerate(branch):
            for ai in range(4):
                rs=[r for r in row['outcomes'] if r['action']==ai]
                if rs:
                    rewards[i,ai]=np.mean([r['return'] for r in rs]);success[i,ai]=np.mean([r['em'] for r in rs]);costs[i,ai]=np.mean([r['cost'] for r in rs])
        targets=(rewards/.1).softmax(-1).detach()
        # Fit auxiliary values for the fixed BC continuation; do not call them calibrated.
        aux=copy.deepcopy(bc)
        for p in aux.parameters():p.requires_grad_(False)
        for layer in [aux.value,aux.success,aux.cost]:
            for p in layer.parameters():p.requires_grad_(True)
        optimizer=torch.optim.AdamW([p for p in aux.parameters() if p.requires_grad],lr=.001)
        with torch.no_grad():basep=distribution(bc(bx,bl)[0],bl,.05).probs;vt=(basep*rewards.masked_fill(~bl,0)).sum(-1)
        for _ in range(100):
            optimizer.zero_grad();_,v,s,c=aux(bx,bl)
            loss=F.mse_loss(v,vt)+F.binary_cross_entropy_with_logits(s[bl],success[bl])+F.mse_loss(c[bl],costs[bl]);loss.backward();optimizer.step()
        for p in aux.parameters():p.requires_grad_(True)
        ce=copy.deepcopy(aux);opt=torch.optim.AdamW(ce.parameters(),lr=.0001)
        for _ in range(100):
            opt.zero_grad();logits,*_=ce(bx,bl);loss=-(targets*logits.log_softmax(-1)).sum(-1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(ce.parameters(),1.);opt.step()
        torch.save(ce.state_dict(),out/'heads/outcome_ce.pt')
        torch.save(aux.state_dict(),out/'heads/bc_aux.pt')
        phase('outcome_training_complete',branch_states=len(branch))
        rl=copy.deepcopy(aux);opt=torch.optim.AdamW(rl.parameters(),lr=.0001);curves=[]
        for iteration in range(3):
            torch.manual_seed(SEED+100+iteration);batch=[];episodes=[];rl.eval()
            for qid in m['train']:
                for rep in range(2):
                    ep,steps,_=episode(qid,rl,f'ppo/iteration{iteration}',f'r{rep}',stochastic=True)
                    returns=returns_for_episode(steps,ep['f1'],m['cost_weight'])
                    for s,ret in zip(steps,returns):
                        if not s['forced']:
                            assert not s['branch_intervention'];batch.append(s|{'return':ret})
                    episodes.append(ep)
                phase('ppo_collection',iteration=iteration,qid=qid)
            tx=torch.stack([s['pack'] for s in batch]).float().to('cuda');tl=torch.stack([s['legal'] for s in batch]).to('cuda')
            ta=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda');ret=torch.tensor([s['return'] for s in batch],device='cuda')
            adv=ret-torch.tensor([s['value'] for s in batch],device='cuda');adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
            with torch.no_grad():
                before=distribution(rl(tx,tl)[0],tl,.05).log_prob(ta)
                assert torch.allclose(before,old,atol=2e-4,rtol=2e-4),'behavior logprob mismatch'
                anchor=distribution(bc(tx,tl)[0],tl,.05).probs
            torch.save(batch,out/f'ppo/iteration{iteration}/on_policy_steps.pt')
            losses=[]
            for _ in range(4):
                for idx in torch.randperm(len(batch),device='cuda').split(32):
                    opt.zero_grad();logits,v,_,_=rl(tx[idx],tl[idx]);dist=distribution(logits,tl[idx],.05)
                    p=dist.probs;kl=(p*(p.clamp_min(1e-9).log()-anchor[idx].clamp_min(1e-9).log())).sum(-1).mean()
                    loss=clipped_policy_loss(dist.log_prob(ta[idx]),old[idx],adv[idx])+.5*F.mse_loss(v,ret[idx])+.02*kl-.01*dist.entropy().mean()
                    loss.backward();torch.nn.utils.clip_grad_norm_(rl.parameters(),1.);opt.step();losses.append(float(loss))
            curves.append({'iteration':iteration,'episodes':len(episodes),'states':len(batch),'train_f1':float(np.mean([e['f1'] for e in episodes])),'mean_loss':float(np.mean(losses))})
            torch.save(rl.state_dict(),out/f'heads/ppo_iteration{iteration}.pt');save(out/'ppo_curve.json',curves);phase('ppo_update_complete',iteration=iteration)
        # No validation-driven choice of PPO iteration or reward coefficient.
        models={'bc':bc,'outcome_ce':ce.eval(),'ppo':rl.eval()};evaluation=[]
        save(out/'frozen_evaluation.json',{'policies':m['test_policies'],'ppo_iteration':2,'model_hashes':{n:sha(out/'heads'/f'{n}.pt') for n in ['outcome_ce']},'test':m['test'],'no_checkpoint_selection':True})
        for group in ['validation','test']:
            phase(group+'_evaluation')
            for qi,qid in enumerate(m[group]):
                policies=m['test_policies'];order=policies[qi%len(policies):]+policies[:qi%len(policies)]
                for policy in order:
                    if policy=='native':
                        reserve();q=raw[qid];rows,vec=corpus(q)
                        ep,_,_=run_episode(q['question'],rows,vec,embedder,reader,prompt,'native','always_generate',deadline)
                        score(ep,q);ep['generated_tokens']=ep['completion_tokens'];ep['forced_prefix_tokens']=0
                        ep['cost_units']=ep['tool_calls']+ep['generated_tokens']/100+ep['prompt_tokens']/10000
                        ep['policy']=policy;charge(ep);save(out/group/f'{policy}_{qid}.json',ep)
                    else:
                        ep,_,_=episode(qid,models.get(policy),group,policy,historical=historical if policy=='historical_bc' else None,native_fast=policy=='native_fast')
                    ep['group']=group;evaluation.append(ep)
                phase(group+'_evaluation',qid=qid)
        summary={}
        for group in ['validation','test']:
            summary[group]={}
            for policy in m['test_policies']:
                es=[e for e in evaluation if e['group']==group and e['policy']==policy]
                summary[group][policy]={'n':len(es),**{k:float(np.mean([e[k] for e in es])) for k in ['f1','em','support_f1','tool_calls','generated_tokens','forced_prefix_tokens','prompt_tokens','seconds','cost_units']},'errors':sum(bool(e['errors']) for e in es),'forced_final':sum(e['forced_final'] for e in es)}
        pairs=[]
        for qid in m['test']:
            es={e['policy']:e for e in evaluation if e['group']=='test' and e['qid']==qid}
            pairs.append({'qid':qid,**{p:es[p]['f1'] for p in m['test_policies']}})
        rng=np.random.default_rng(SEED);cis={}
        for p in ['ppo','outcome_ce','native_fast']:
            d=np.array([x[p]-x['bc'] for x in pairs]);boot=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
            cis[p+'_minus_bc']={'mean':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'wins':int((d>0).sum()),'losses':int((d<0).sum())}
        save(out/'summary.json',{'policies':summary,'paired':pairs,'exploratory_ci':cis,'single_seed':True,'controlled_latency':False,'success_probabilities_calibrated':False,'worker_counts':counts})
        phase('complete',training_complete=True,evaluation_complete=True)
    except Exception as e:
        phase('failed_or_budget_limited',error=type(e).__name__+': '+str(e));raise
    finally:
        if embedder:embedder.close()

if __name__=='__main__':main()
