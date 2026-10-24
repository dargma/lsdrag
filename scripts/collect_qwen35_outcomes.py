"""Fresh9B branch outcomes on every controllable prefix of frozen48 rollins."""
import argparse,copy,fcntl,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_policy import rollout
from lsdrag.text_decision_benchmark import canonical
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);args=ap.parse_args();spec=read(args.spec)
    for p,h in spec['hashes'].items():assert sha(p)==h,p
    assert spec['mode'] in ['all_actions','stop_pairs']
    train=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json');assert spec['qids']==train['outcome_train48']==train['stop_train48']
    lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'results'/spec['run_name'];out.mkdir(exist_ok=False);save(out/'spec.json',spec)
    start=time.monotonic();counts=dict(episodes=0,generated=0,inputs=0,states=0);embed=None
    def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**counts,**kw))
    phase('loading');torch.set_num_threads(2)
    raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open())};assert set(spec['qids'])<=raw.keys()
    sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
    try:
        head=StateHead(dim=4096,inputs='multi',body='mlp');head.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location='cpu'));head=head.to('cuda').eval()
        reader=Qwen35Reader(spec['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder()
        pool=Path(spec['corpus']);rows=[json.loads(x) for x in (pool/'documents.jsonl').open()];vectors=np.load(pool/'vectors.npy',mmap_mode='r')
        prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();deadline=time.perf_counter()+spec['max_seconds'];records=[]
        for qi,qid in enumerate(spec['qids']):
            q=raw[qid];answers=[q['answer']]+q['answer_aliases']
            def episode(tag,initial=None,first=None):
                assert counts['episodes']<spec['max_episodes'] and counts['generated']+4608<spec['max_generated']
                e,steps,_=rollout(reader,q['question'],rows,vectors,embed,prompt,head,deadline,stochastic=False,epsilon=0.,initial=initial,first_action=first,state_storage_dtype=torch.float32)
                e.update(qid=qid,tag=tag,f1=max(compute_f1(a,e['answer']) for a in answers),em=max(compute_exact(a,e['answer']) for a in answers))
                save(out/'episodes'/f'{qid}_{tag}.json',e);counts['episodes']+=1;counts['generated']+=e['generated_tokens'];counts['inputs']+=e['prompt_tokens'];return e,steps
            parent,steps=episode('collector');prefills=[u for u in parent['reader_calls'] if u.get('pack_slots')];assert len(prefills)==len(steps)
            for turn,state in enumerate(steps):
                if state['forced']:continue
                messages=[{'role':'system','content':prompt},{'role':'user','content':q['question']}]
                history=copy.deepcopy(parent['trajectory'][:turn])
                for i,step in enumerate(history):
                    tc={'id':f'call_{i}','type':'function','function':{'name':step['tool_name'],'arguments':canonical(step['arguments'])}}
                    messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]},{'role':'tool','tool_call_id':tc['id'],'content':canonical(step['result'])}])
                prefix=hashlib.sha256(canonical(messages).encode()).hexdigest();assert prefix==prefills[turn]['messages_sha256']
                legal=state['legal'];probs=state['policy_probs'];cont=int(torch.tensor(probs[:3]).masked_fill(~legal[:3],-1).argmax())
                actions=torch.where(legal)[0].tolist() if spec['mode']=='all_actions' else [cont,3]
                samples=[]
                for action in actions:
                    e,branch=episode(f't{turn}_a{action}_r0',dict(messages=messages,history=history),action)
                    assert e['reader_calls'][0]['messages_sha256']==prefix and branch[0]['action']==action
                    assert torch.equal(branch[0]['legal'],legal)
                    assert torch.allclose(branch[0]['pack'],state['pack'],atol=.04,rtol=.02)
                    samples.append(dict(action=action,rep=0,f1=e['f1'],em=e['em'],cost=e['cost_units'],generated=e['generated_tokens'],inputs=e['prompt_tokens'],errors=e['errors']))
                records.append(dict(qid=qid,turn=turn,pack=state['pack'],legal=legal,prefix_sha256=prefix,reference_probs=probs,continue_action=cont,samples=samples))
                counts['states']=len(records);torch.save(records,out/'states.pt');phase('collection',questions_started=qi+1,qid=qid,turn=turn)
        save(out/'summary.json',dict(states=len(records),questions=len(spec['qids']),mode=spec['mode'],tensor_sha256=sha(out/'states.pt'),errors=sum(bool(s['errors']) for r in records for s in r['samples'])))
        phase('complete')
    except BaseException as e:phase('failed',error=repr(e));raise
    finally:
        if embed:embed.close()
if __name__=='__main__':main()
