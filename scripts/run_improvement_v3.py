"""Merged train24 action/read heads, validation8 ablation, frozen fresh20."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lsdrag.text_integrated import ACTIONS,slots_for,run_episode
from lsdrag.text_integrated_aligned import AlignedReader
from lsdrag.text_decision_benchmark import public_paragraphs,canonical
from lsdrag.v11_embedding_worker import V11ProcessEmbedder

sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def train(records,features,contract):
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261016)
    coverage={a:len({r['qid'] for r in records if r['action']==a}) for a in ACTIONS}
    read=[r for r in records if r['action']=='read_chunk']
    gate={'profile':'train24-action-read-v3','questions':len({r['qid'] for r in records}),'per_action':coverage,'read_reuse_qids':len({r['qid'] for r in read if r['slot']>0}),'read_generate_qids':len({r['qid'] for r in read if r['slot']==0}),'read_slot_counts':{str(i):sum(r['slot']==i for r in read) for i in range(5)},'minimums':{'questions':24,'per_action':5,'read_reuse':5,'read_generate':5}}
    gate['passed']=gate['questions']==24 and min(coverage.values())>=5 and gate['read_reuse_qids']>=5 and gate['read_generate_qids']>=5
    if not gate['passed']:return None,{'gate':gate}
    x=np.asarray(features,dtype=np.float32);mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    tx=torch.from_numpy((x-mean)/scale);y=torch.tensor([r['action_index'] for r in records]);sy=torch.tensor([r['slot'] for r in records]);idx=y==2
    legal=torch.tensor([[any(o['action']==a for o in r['options']) for a in ACTIONS] for r in records]);mask=torch.tensor([[i<len(slots_for(r['options'],'read_chunk')) for i in range(5)] for r in records])[idx]
    action=torch.nn.Linear(x.shape[1],4);slot=torch.nn.Linear(x.shape[1],5);opt=torch.optim.AdamW(list(action.parameters())+list(slot.parameters()),lr=.001,weight_decay=.01);losses=[]
    for _ in range(100):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(action(tx).masked_fill(~legal,-torch.inf),y)+torch.nn.functional.cross_entropy(slot(tx)[idx].masked_fill(~mask,-torch.inf),sy[idx]);loss.backward();opt.step();losses.append(float(loss.detach()))
    def fold(layer):
        w=layer.weight.detach()/torch.from_numpy(scale)[None,:];b=layer.bias.detach()-w@torch.from_numpy(mean)
        torch.testing.assert_close(layer(tx),torch.from_numpy(x)@w.T+b,atol=1e-4,rtol=1e-4)
        return w,b
    aw,ab=fold(action);rw,rb=fold(slot);sw=torch.zeros((3,5,x.shape[1]));sb=torch.zeros((3,5));sw[2]=rw;sb[2]=rb
    # Search always chooses GENERATE; no search slot training.
    sb[:2,0]=1
    ck={'format':'text-action-read-v3','contract':contract,'actions':ACTIONS,'action_weight':aw,'action_bias':ab,'slot_weight':sw,'slot_bias':sb,'train_qids':sorted({r['qid'] for r in records}),'seed':20261016,'epochs':100,'search_slots':'GENERATE_ONLY','read_slots':'GENERATE,observed1..4'}
    return ck,{'gate':gate,'loss':losses,'trained':True,'train_feature_normalization_only':True,'recovery_added':False,'note':'Missing individual read slots are reported, not evidence of full slot coverage.'}


def fixed_metrics(records,features,ck):
    import torch
    x=torch.from_numpy(np.asarray(features,dtype=np.float32));scores=x@ck['action_weight'].T+ck['action_bias'];mask=torch.tensor([[any(o['action']==a for o in r['options']) for a in ACTIONS] for r in records]);pred=scores.masked_fill(~mask,-torch.inf).argmax(-1).tolist();cm=np.zeros((4,4),dtype=int)
    for r,p in zip(records,pred):cm[r['action_index'],p]+=1
    f=[]
    for i in range(4):
        d=int(cm[i,:].sum()+cm[:,i].sum());f.append(2*int(cm[i,i])/d if d else 0.)
    return {'n':len(records),'confusion':cm.tolist(),'action_order':ACTIONS,'macro_f1':float(np.mean(f)),'per_action_f1':f,'premature_final':sum(p==3 and r['action_index']!=3 for r,p in zip(records,pred)),'missing_true_classes':[ACTIONS[i] for i in range(4) if cm[i,:].sum()==0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch,fcntl
    root=Path('/content/local_fast/lsdrag-text-v1');repo=Path(__file__).resolve().parents[1];parent=root/'results/text_integrated_20261016_v5_fast_only';check=root/'results/improvement_v3_aligned_20261017'
    lock=(root/'improvement_v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not json.loads((check/'summary.json').read_text())['gate_passed']:raise ValueError('Aligned prerequisite failed')
    split=json.loads((repo/'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
    original=json.loads((parent/'manifest.json').read_text())
    for name in ['src/lsdrag/text_integrated.py','src/lsdrag/text_only_reader.py','prompts/text_rag_integrated_v5.txt']:
        if sha(repo/name)!=original['source_sha256'][name]:raise ValueError('Feature origin contract changed: '+name)
    config={'split':split,'profile':'v3-action-read','training_seed':20261016,'epochs':100,'frozen_backbone':True,'loss':'action CE + read-state slot CE','learning_rate':.001,'weight_decay':.01,'validation_episodes':16,'fresh_episodes':40,'recovery':'optional not used in this bounded run','teacher_manifest_sha256':sha(parent/'manifest.json'),'aligned_gate_sha256':sha(check/'summary.json'),'max_output_tokens':512,'context_limit':16384,'worker_minutes':45,'source_sha256':{n:sha(repo/n) for n in ['scripts/run_improvement_v3.py','src/lsdrag/text_integrated_aligned.py','src/lsdrag/text_integrated.py']},'train24_gate':'exact24 natural qids; each action>=5 qids; read reuse andGENERATE each>=5 qids; individual slot absence reported'}
    contract=hashlib.sha256(canonical(config).encode()).hexdigest()
    def save(name,value):
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
    started=time.perf_counter();deadline=started+2700
    def phase(name,**kw):save('status.json',{'phase':name,'pid':os.getpid(),'worker_seconds':time.perf_counter()-started,**kw});print(name,kw,flush=True)
    save('manifest.json',config);phase('cpu_training')
    sets={}
    for name in ['train','validation']:
        rs=[];fs=[]
        for q in split[name]:
            rec=json.loads((parent/'teacher'/f'{q}.json').read_text())['records'];feat=np.load(parent/'teacher'/f'{q}.npz')['features'];assert len(rec)==len(feat)
            rs.extend(rec);fs.extend(feat)
        sets[name]=(rs,fs)
    ck,training=train(*sets['train'],contract);save('training.json',training)
    if ck is None:phase('coverage_nogo');return
    torch.save(ck,a.output/'head.pt');save('fixed_validation.json',fixed_metrics(*sets['validation'],ck))
    raw={}
    for name in ['train','dev']:
        with (root/f'source/musique/musique_ans_v1.0_{name}.jsonl').open() as f:raw[name]=[json.loads(line) for line in f]
    byid={q['id']:q for q in raw['train']}
    sys.path.insert(0,str(root/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
    def score(ep,q):
        aliases=[q['answer']]+q['answer_aliases'];gold={f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']};pred=set(ep['support_ids']);overlap=len(gold&pred)
        ep.update(qid=q['id'],f1=max(compute_f1(ans,ep['answer']) for ans in aliases),em=max(compute_exact(ans,ep['answer']) for ans in aliases),support_f1=2*overlap/(len(gold)+len(pred)) if gold or pred else 0.)
        ep['decoder_forward_count']=sum(max(0,c['usage'].get('generated_tokens',0)-1)+int(c['usage'].get('forced_prefix_tokens',0)>0) for c in ep['reader_calls'] if c['usage'].get('generated_tokens',0)>0)
        ep['prefill_count']=sum(c['usage'].get('full_prefill_count',0) for c in ep['reader_calls']);ep['forced_prefix_tokens']=sum(c['usage'].get('forced_prefix_tokens',0) for c in ep['reader_calls']);return ep
    embedder=None
    try:
        phase('validation_loading');embedder=V11ProcessEmbedder();reader=AlignedReader(original['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));reader.checkpoint={k:ck[k].to('cuda') for k in ['action_weight','action_bias','slot_weight','slot_bias']};prompt=(repo/'prompts/text_rag_integrated_v5.txt').read_text();cache={}
        def corpus(q):
            if q['id'] not in cache:
                rows=public_paragraphs(q);path=parent/'index'/f'{q["id"]}.npy'
                vec=np.load(path) if path.exists() else embedder.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
                cache[q['id']]=(rows,vec)
            return cache[q['id']]
        def episode(q,policy,folder):
            if time.perf_counter()>deadline:raise TimeoutError('45min worker cap')
            rows,vec=corpus(q);ep,_,_=run_episode(q['question'],rows,vec,embedder,reader,prompt,'native' if policy=='native' else 'head','reuse' if policy=='read_selection' else 'always_generate',deadline)
            ep=score(ep,q);ep['policy']=policy;save(f'{folder}/{policy}/{q["id"]}.json',ep);print(folder,q['id'],policy,ep['f1'],ep['errors'],flush=True);return ep
        def aggregate(episodes):
            return {p:{'n':len(es),**{k:float(np.mean([e[k] for e in es])) for k in ['f1','em','support_f1','argument_tokens','completion_tokens','prompt_tokens','tool_calls','reused_argument_calls','prefill_count','forced_prefix_tokens','decoder_forward_count']},'errors':sum(bool(e['errors']) for e in es)} for p in sorted({e['policy'] for e in episodes}) if (es:=[e for e in episodes if e['policy']==p])}
        phase('validation8');validation=[]
        for qid in split['validation']:
            for policy in ['always_generate','read_selection']:validation.append(episode(byid[qid],policy,'validation'))
        vs=aggregate(validation);base=vs['always_generate'];read=vs['read_selection'];adopt=read['reused_argument_calls']>0 and read['f1']>=base['f1'] and read['em']>=base['em'] and read['errors']==0
        primary='read_selection' if adopt else 'always_generate';save('validation_summary.json',{'policies':vs,'read_adopted':adopt,'primary':primary,'exploratory':True})
        registry=json.loads((repo/'lab/registry/text_integrated_20261016/split.json').read_text());exposed=set(registry['train'])|set(registry['dev'])|set(json.loads((repo/'lab/registry/text_integrated_20261016/next_holdout_reserved.json').read_text())['qids'])
        families={d['id'] for q in raw['train'] if q['id'] in exposed for d in q['question_decomposition']}
        eligible=[q for q in raw['dev'] if q['id'] not in exposed and q['answerable'] and len(q['question_decomposition'])==2 and not {d['id'] for d in q['question_decomposition']}&families]
        fresh=sorted(eligible,key=lambda q:hashlib.sha256(('improvement_v3_20261017:'+q['id']).encode()).hexdigest())[:20];assert len(fresh)==20
        save('frozen_primary.json',{'primary':primary,'reason':'read selection passes validation gate' if adopt else 'read selection not adopted; preserve always-generate','qids':[q['id'] for q in fresh],'excluded_qids':sorted(exposed),'head_sha256':sha(a.output/'head.pt'),'source_sha256':config['source_sha256'],'prompt_sha256':sha(repo/'prompts/text_rag_integrated_v5.txt'),'model':original['model'],'contract':contract,'frozen_before_fresh_outcomes':True})
        phase('fresh20',primary=primary);evaluation=[]
        for i,q in enumerate(fresh):
            for policy in (['native',primary] if i%2==0 else [primary,'native']):evaluation.append(episode(q,policy,'evaluation'))
        summary=aggregate(evaluation);pairs=[]
        for q in fresh:
            es={e['policy']:e for e in evaluation if e['qid']==q['id']};pairs.append({'qid':q['id'],'native_f1':es['native']['f1'],'primary_f1':es[primary]['f1'],'f1_delta':es[primary]['f1']-es['native']['f1']})
        deltas=np.array([p['f1_delta'] for p in pairs]);rng=np.random.default_rng(20261016);boot=deltas[rng.integers(0,20,size=(2000,20))].mean(1)
        save('summary.json',{'policies':summary,'primary':primary,'pairs':pairs,'f1_difference':float(deltas.mean()),'exploratory_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'wins':int((deltas>0).sum()),'losses':int((deltas<0).sum()),'ties':int((deltas==0).sum()),'controlled_latency_claim':False,'quality_noninferiority_claim':False,'recovery_added':False});phase('complete',training_completed=True,validation_episodes=16,evaluation_episodes=40)
    except Exception as error:phase('failed_or_budget_limited',error=type(error).__name__+': '+str(error));raise
    finally:
        if embedder:embedder.close()
if __name__=='__main__':main()
