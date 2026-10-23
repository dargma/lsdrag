"""Shared-data first/middle/late-state policy improvement, bounded pilot2."""
import argparse,copy,hashlib,json,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout
from lsdrag.decision_policy_followup import tempered_initialization,state_snapshots,fit_improvement
from lsdrag.text_integrated import ACTIONS,options_for,run_episode
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save

PREVIOUS=ROOT/'results/decision_policy_20261023_pilot1'
REG=REPO/'lab/registry/decision_policy_pilot2_20261023'
SPECS={'mlp_pg_quality':('mlp','return_pg',0.),'mlp_pg_cost':('mlp','return_pg',.03),
       'mlp_ce_quality':('mlp','outcome_ce',0.),'mlp_ce_cost':('mlp','outcome_ce',.03),
       'linear_pg_cost':('linear','return_pg',.03),'linear_ce_cost':('linear','outcome_ce',.03)}


def prepare():
    old=json.loads((PREVIOUS/'manifest.json').read_text());audit=json.loads((REPO/'lab/reports/decision_policy_20261023/pilot1_train_audit.json').read_text())
    exposed=set()
    for folder in [REPO/'lab',ROOT/'results']:
        for path in folder.rglob('*.json'):exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',path.read_text()))
    data={n:[json.loads(x) for x in (ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl').read_text().splitlines()] for n in ['train','dev']}
    families={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']}
    eligible=[q for q in data['dev'] if q['answerable'] and len(q['question_decomposition'])==2 and q['id'] not in exposed and not {d['id'] for d in q['question_decomposition']}&families]
    test=[];used=set()
    for q in sorted(eligible,key=lambda q:hashlib.sha256(('policy-pilot2:'+q['id']).encode()).hexdigest()):
        f={d['id'] for d in q['question_decomposition']}
        if not f&used:test.append(q['id']);used.update(f)
        if len(test)==40:break
    assert len(test)==40
    source=['src/lsdrag/decision_policy_followup.py','scripts/run_decision_policy_followup.py','src/lsdrag/decision_policy.py','scripts/run_decision_policy_pilot.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']
    m={'profile':'decision_policy_pilot2_20261023','seed':20261024,'train':old['train'],'validation':old['validation'],'test':test,
       'excluded_qids':sorted(exposed),'eligible_fresh':len(eligible),'temperature':audit['chosen_temperature'],'model':old['model'],
       'reference':'round0 tempered BC; round1 round0 mlp_pg_quality; every suffix controlled by same frozen reference',
       'rounds':2,'snapshots':'first/middle/last controllable state from one exploratory trajectory/question; cap forced FINAL excluded',
       'branch_repeats':2,'branch_cap':1152,'collectors':48,'specs':SPECS,'epochs':100,'lr':.001,'anchor_KL':.02,'entropy_coef':.01,
       'state_weighting':'equal questions; inverse count of selected states within question',
       'method':'all-action Monte Carlo policy-improvement surrogate, NOT PPO or unbiased on-policy policy gradient',
       'data_match':'all6 learners receive identical state/action/sample outcomes at each round; shared reference distribution favors neither separate collection budget',
       'limits':{'seconds_excluding_load':3600,'generated_tokens':1000000,'input_tokens':16000000,'max_tool_calls':8,'max_output_tokens':512,'context_limit':16384},
       'primary':'mlp_pg_cost vs mlp_ce_cost and historical_bc; no val-selected checkpoint or cost coefficient',
       'evaluation_policies':['native','historical_bc']+list(SPECS),'single_seed':True,'test_frozen_before_outcomes':True,
       'initial_checkpoint_sha256':{n:sha(PREVIOUS/'heads'/f'{n}.pt') for n in ['last_mlp','last_linear']},
       'source_sha256':{n:sha(REPO/n) for n in source}}
    REG.mkdir(parents=True,exist_ok=True)
    if (REG/'manifest.json').exists():raise FileExistsError('frozen manifest exists')
    save(REG/'manifest.json',m);print(REG/'manifest.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.prepare:prepare();return
    m=json.loads((REG/'manifest.json').read_text());assert all(sha(REPO/n)==h for n,h in m['source_sha256'].items())
    assert all(sha(PREVIOUS/'heads'/f'{n}.pt')==h for n,h in m['initial_checkpoint_sha256'].items())
    out=a.output;out.mkdir(parents=True,exist_ok=False);save(out/'manifest.json',m)
    import fcntl
    lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(m['seed']);start=time.perf_counter();deadline=None;embedder=None
    counts={'episodes':0,'branches':0,'generated':0,'inputs':0}
    def phase(name,**kw):
        s={'phase':name,'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-start,**counts,**kw};save(out/'status.json',s);print(json.dumps(s),flush=True)
    def reserve():
        caps=m['limits']
        if time.perf_counter()>deadline or counts['generated']+4608>caps['generated_tokens'] or counts['inputs']+148032>caps['input_tokens']:raise TimeoutError('Bounded worker cap')
    def charge(ep):
        counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens']
        assert counts['generated']<=m['limits']['generated_tokens'] and counts['inputs']<=m['limits']['input_tokens']
    wanted=set(m['train']+m['validation']+m['test']);raw={}
    for name in ['train','dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q=json.loads(line)
            if q['id'] in wanted:raw[q['id']]=q
    sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
    def score(ep,q):
        aliases=[q['answer']]+q['answer_aliases'];gold={f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']};pred=set(ep['support_ids'])
        ep.update(qid=q['id'],em=max(compute_exact(x,ep['answer']) for x in aliases),f1=max(compute_f1(x,ep['answer']) for x in aliases),support_f1=2*len(gold&pred)/max(1,len(gold)+len(pred)))
        return ep
    try:
        phase('loading');reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embedder=V11ProcessEmbedder()
        prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();deadline=time.perf_counter()+3600;cache={}
        hist=torch.load(ORIGIN/'head.pt',weights_only=True,map_location='cuda');historical={k:hist[k] for k in ['action_weight','action_bias']}
        anchors={}
        for arch,n in [('mlp','last_mlp'),('linear','last_linear')]:
            h=PolicyHead(2560,arch).cuda();h.load_state_dict(torch.load(PREVIOUS/'heads'/f'{n}.pt',weights_only=True,map_location='cuda'));anchors[arch]=tempered_initialization(h,m['temperature']).eval()
        learners={name:copy.deepcopy(anchors[spec[0]]) for name,spec in SPECS.items()}
        def corpus(q):
            if q['id'] not in cache:
                rows=public_paragraphs(q);p=PARENT/'index'/f'{q["id"]}.npy';vec=np.load(p) if p.exists() else embedder.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows]);cache[q['id']]=(rows,vec)
            return cache[q['id']]
        def episode(qid,head,folder,tag,**kw):
            reserve();q=raw[qid];rows,vec=corpus(q)
            ep,steps,_=rollout(reader,q['question'],rows,vec,embedder,prompt,head,deadline,epsilon=0.,**kw)
            score(ep,q);ep['policy']=tag
            for d in ep['decisions']:
                d['selected_action']=ACTIONS[d['action']];d['action_probabilities']=dict(zip(ACTIONS,d['policy_probs']))
                d['probability_semantics']='action-selection distribution; not terminal-success probability'
            charge(ep);save(out/folder/f'{tag}_{qid}.json',ep);return ep,steps
        statistics=[]
        for round_id in range(2):
            reference=copy.deepcopy(anchors['mlp'] if round_id==0 else learners['mlp_pg_quality']).eval();rows=[]
            torch.save(reference.state_dict(),out/f'reference_round{round_id}.pt')
            for qi,qid in enumerate(m['train']):
                torch.manual_seed(m['seed']+round_id*1000+qi)
                collector,_=episode(qid,reference,f'round{round_id}/collectors','reference',stochastic=True)
                snaps=state_snapshots(raw[qid]['question'],prompt,collector['trajectory'])
                for si,snapshot in enumerate(snaps):
                    opts=options_for(raw[qid]['question'],snapshot['history']);legal=[any(o['action']==a for o in opts) for a in ACTIONS];samples=[]
                    for ai,ok in enumerate(legal):
                        if not ok:continue
                        for rep in range(2):
                            if counts['branches']>=1152:raise RuntimeError('branch cap')
                            # Common suffix RNG seeds for paired action comparisons.
                            torch.manual_seed(m['seed']+round_id*100000+qi*1000+snapshot['turn']*10+rep)
                            ep,steps=episode(qid,reference,f'round{round_id}/branches',f't{snapshot["turn"]}_a{ai}_r{rep}',stochastic=True,initial=snapshot,first_action=ai)
                            counts['branches']+=1;samples.append({'action':ai,'f1':ep['f1'],'em':ep['em'],'cost':ep['cost_units'],'calls':ep['tool_calls'],'errors':ep['errors']})
                    rows.append({'qid':qid,'turn':snapshot['turn'],'pack':steps[0]['pack'],'legal':steps[0]['legal'],'samples':samples,'reference_round':round_id})
                phase('counterfactual_collection',round=round_id,qid=qid,states=len(rows))
            torch.save(rows,out/f'round{round_id}/counterfactual_data.pt')
            x=torch.stack([r['pack'] for r in rows]).float().cuda();legal=torch.stack([r['legal'] for r in rows]).cuda();quality=torch.zeros(len(rows),4,device='cuda');cost=torch.zeros_like(quality)
            for i,row in enumerate(rows):
                for ai in range(4):
                    samples=[s for s in row['samples'] if s['action']==ai]
                    if samples:quality[i,ai]=np.mean([s['f1'] for s in samples]);cost[i,ai]=np.mean([s['cost'] for s in samples])
            report={'round':round_id,'states':len(rows),'turn_histogram':{str(t):sum(r['turn']==t for r in rows) for t in range(8)},'late_states_ge3':sum(r['turn']>=3 for r in rows),'models':{}}
            for name,(arch,objective,lam) in SPECS.items():
                q=quality-lam*cost
                learners[name],curve=fit_improvement(learners[name],anchors[arch],x,legal,q,[r['qid'] for r in rows],objective)
                (out/'heads').mkdir(exist_ok=True);torch.save(learners[name].state_dict(),out/'heads'/f'{name}_round{round_id}.pt')
                save(out/f'round{round_id}/training_{name}.json',curve);report['models'][name]={'loss_final':curve[-1],'data_sha256':sha(out/f'round{round_id}/counterfactual_data.pt'),'cost_weight':lam,'objective':objective}
            statistics.append(report);save(out/'collection_training.json',statistics);phase('policy_update_complete',round=round_id,states=len(rows))
        save(out/'frozen_evaluation.json',{'policies':m['evaluation_policies'],'test':m['test'],'round':1,'head_sha256':{n:sha(out/'heads'/f'{n}_round1.pt') for n in SPECS},'no_validation_model_selection':True})
        results=[]
        for group in ['validation','test']:
            for i,qid in enumerate(m[group]):
                policies=m['evaluation_policies'];policies=policies[i%len(policies):]+policies[:i%len(policies)]
                for name in policies:
                    if name=='native':
                        reserve();q=raw[qid];r,v=corpus(q);ep,_,_=run_episode(q['question'],r,v,embedder,reader,prompt,'native','always_generate',deadline);score(ep,q);ep.update(policy=name,generated_tokens=ep['completion_tokens'],forced_prefix_tokens=0)
                        ep['cost_units']=ep['tool_calls']+ep['generated_tokens']/100+ep['prompt_tokens']/10000;charge(ep);save(out/group/f'{name}_{qid}.json',ep)
                    else:ep,_=episode(qid,learners.get(name),group,name,historical=historical if name=='historical_bc' else None)
                    ep['group']=group;results.append(ep)
                phase(group+'_evaluation',qid=qid)
        summary={}
        for group in ['validation','test']:
            summary[group]={}
            for name in m['evaluation_policies']:
                eps=[e for e in results if e['group']==group and e['policy']==name]
                summary[group][name]={'n':len(eps),**{k:float(np.mean([e[k] for e in eps])) for k in ['f1','em','support_f1','tool_calls','generated_tokens','prompt_tokens','forced_prefix_tokens','seconds','cost_units']},'errors':sum(bool(e['errors']) for e in eps),'forced_final':sum(e['forced_final'] for e in eps)}
        comparisons={};rng=np.random.default_rng(m['seed'])
        for left,right in [('mlp_pg_cost','historical_bc'),('mlp_pg_cost','mlp_ce_cost'),('mlp_pg_cost','mlp_pg_quality'),('linear_pg_cost','linear_ce_cost')]:
            pairs=[]
            for qid in m['test']:
                eps={e['policy']:e for e in results if e['group']=='test' and e['qid']==qid};pairs.append({'qid':qid,'f1_delta':eps[left]['f1']-eps[right]['f1'],'calls_delta':eps[left]['tool_calls']-eps[right]['tool_calls']})
            d=np.array([v['f1_delta'] for v in pairs]);bootstrap=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(-1)
            comparisons[left+'_minus_'+right]={'f1_delta':float(d.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'pairs':pairs}
        save(out/'summary.json',{'policies':summary,'comparisons':comparisons,'counts':counts,'single_seed':True,'controlled_latency':False,'method':'shared-data all-action return-based approximate policy improvement; not PPO','success_probabilities_calibrated':False})
        phase('complete',training_complete=True,evaluation_complete=True)
    except Exception as e:phase('failed_or_budget_limited',error=type(e).__name__+': '+str(e));raise
    finally:
        if embedder:embedder.close()

if __name__=='__main__':main()
