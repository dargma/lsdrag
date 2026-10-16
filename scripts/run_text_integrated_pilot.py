"""Bounded compact-reference pilot: train4 cost screen, teacher32/48, heads, dev10."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np

from lsdrag.io import sha256
from lsdrag.text_decision_benchmark import TextCorpus, TOOLS, canonical, public_paragraphs
from lsdrag.text_integrated import (ACTIONS, StrictReader, normalize_args, options_for,
    run_episode, selection_messages, slots_for, train_heads)
from lsdrag.v11_embedding_worker import V11ProcessEmbedder


def cost_summary(rows):
    """Question-cluster bootstrap; errors are never silently removed as fast cases."""
    policies=['integrated','always_generate']
    result={}
    for policy in policies:
        pairs=[]; failures=0
        for row in rows:
            baseline=row['conditions']['native'];student=row['conditions'][policy]
            failures+=int(bool(baseline['failure'] or student['failure']))
            pairs.append((row['qid'],row['bucket'],baseline['seconds'],student['seconds']))
        buckets={}
        for bucket in ['all','reuse','generate','FINAL']:
            subset=[p for p in pairs if bucket=='all' or p[1]==bucket]
            qids=sorted({p[0] for p in subset})
            if not subset:
                buckets[bucket]={'n':0,'passed':False};continue
            matrix=np.array([[sum(p[2] for p in subset if p[0]==q),
                              sum(p[3] for p in subset if p[0]==q)] for q in qids])
            rng=np.random.default_rng(20261016)
            samples=matrix[rng.integers(0,len(qids),size=(2000,len(qids)))].sum(1)
            ratios=samples[:,0]/samples[:,1]
            lo,hi=np.quantile(ratios,[.025,.975])
            buckets[bucket]={'n':len(subset),'qids':len(qids),'speedup':float(matrix[:,0].sum()/matrix[:,1].sum()),
                'ci95':[float(lo),float(hi)],'passed':bool(len(qids)>=2 and lo>1.0)}
        result[policy]={'failures':failures,'buckets':buckets,
            'passed':failures==0 and all(v['passed'] for v in buckets.values())}
    return {'policies':result,'passed':result['integrated']['passed'],
        'scope':'Train4 interleaved matched-state/oracle-action cost feasibility only; not learned quality or controlled latency.',
        'same_states_and_actions':True,'generated_argument_text_may_differ':True,
        'shared_gpu':True,'controlled_latency_claim':False,'head_is_trained':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minutes',type=float,default=58)
    args=parser.parse_args()
    root=Path('/content/local_fast/lsdrag-text-v1');repo=Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root.resolve()):parser.error('Run output must be local-fast')
    if not 0<args.minutes<=58:parser.error('Reserve2min for current functional smoke within60min cap')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.perf_counter()
    spent=json.loads((args.output/'status.json').read_text()).get('worker_seconds',0) if (args.output/'status.json').exists() else 0
    deadline=started+args.minutes*60-spent
    def save(name,value):
        path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
    def phase(name,**details):
        value={'phase':name,'pid':os.getpid(),'run':str(args.output),'updated_epoch':time.time(),
               'worker_seconds':spent+time.perf_counter()-started,**details}
        save('status.json',value);print(json.dumps(value),flush=True)
        p=repo/'memory/EXPERIMENT_STATUS.json';old=json.loads(p.read_text());old.update(value)
        p.write_text(json.dumps(old,indent=2)+'\n')
        header='# Compact integrated pilot — '+name+'\n\n'+json.dumps(value)+'\n\n'
        for name in ['CURRENT.md','HEAD_EXPERIMENT.md']:
            p=repo/'memory'/name;p.write_text(header+p.read_text())
        p=repo/'memory/SESSION_LOG.md';p.write_text(p.read_text()+'\n'+header)
    def backup():
        archive=root/(args.output.name+'.tar.gz')
        with tarfile.open(archive,'w:gz') as target:target.add(args.output,arcname=args.output.name)
        destination=Path('/content/drive/MyDrive/lsdrag-backups/20261016/text_integrated_20261016')/archive.name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive,destination)
        if sha256(destination)!=sha256(archive):raise RuntimeError('Backup hash mismatch')
        save('backup.json',{'path':str(destination),'sha256':sha256(archive)})
    def remaining():
        if time.perf_counter()>=deadline:raise TimeoutError('Cumulative worker budget reached')
    source=['scripts/run_text_integrated_pilot.py','src/lsdrag/text_integrated.py',
        'src/lsdrag/text_only_reader.py','src/lsdrag/text_decision_benchmark.py','src/lsdrag/tool_registry.py',
        'src/lsdrag/v11_embedding_worker.py','src/lsdrag/embeddings.py','prompts/text_rag_integrated_v5.txt']
    split=json.loads((repo/'lab/registry/text_integrated_20261016/split.json').read_text())
    holdout=json.loads((repo/'lab/registry/text_integrated_20261016/next_holdout_reserved.json').read_text())['qids']
    model=root/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
    manifest={'experiment':'text-integrated-implicit-v5','source_sha256':{p:sha256(repo/p) for p in source},
        'train_qids':split['train'],'holdout_qids':holdout,'model':str(model),'thinking':False,'actions':ACTIONS,
        'tools':TOOLS,'max_calls':8,'max_output_tokens':512,'context_limit':16384,'seed':20261016,
        'cost_screen_repeats':3,'cost_screen_states':'all natural train4 states including FINAL',
        'data_sha256':{s:sha256(root/f'source/musique/musique_ans_v1.0_{s}.jsonl') for s in ['train','dev']},
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','lm-format-enforcer']}}
    # JSON-normalized tuples so resume comparisons preserve identity.
    manifest=json.loads(json.dumps(manifest));contract=hashlib.sha256(canonical(manifest).encode()).hexdigest()
    if (args.output/'manifest.json').exists() and json.loads((args.output/'manifest.json').read_text())!=manifest:
        raise ValueError('Changed contract; choose a fresh run identity')
    save('manifest.json',manifest);save('launch.json',{'pid':os.getpid(),'argv':sys.argv,'contract':contract})
    raw={}
    for name in ['train','dev']:
        with (root/f'source/musique/musique_ans_v1.0_{name}.jsonl').open() as stream:
            raw[name]={r['id']:r for line in stream if (r:=json.loads(line))['id'] in set(split['train'] if name=='train' else holdout)}
    family=lambda rows:{d['id'] for row in rows for d in row['question_decomposition']}
    if family(raw['train'].values())&family(raw['dev'].values()):raise ValueError('Question family overlap')
    expected={'train':'83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a',
              'dev':'15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b'}
    if manifest['data_sha256']!=expected:raise ValueError('Dataset identity mismatch')
    sys.path.insert(0,str(root/'source/musique_scorer'))
    from metrics.answer import compute_exact,compute_f1
    save('scorer_provenance.json',{str(p.relative_to(root/'source/musique_scorer')):sha256(p) for p in (root/'source/musique_scorer').rglob('*.py')})
    prompt=(repo/'prompts/text_rag_integrated_v5.txt').read_text()
    embedder=None;records=[];features=[];episodes=[];corpora={}
    def corpus(q):
        if q['id'] not in corpora:
            remaining();rows=public_paragraphs(q);path=args.output/'index'/f'{q["id"]}.npy';path.parent.mkdir(exist_ok=True)
            if path.exists():vectors=np.load(path,allow_pickle=False)
            else:
                vectors=embedder.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows]);np.save(path,vectors)
            if vectors.shape!=(len(rows),768) or not np.isfinite(vectors).all():raise ValueError('Invalid corpus vectors')
            corpora[q['id']]=(rows,vectors)
        return corpora[q['id']]
    def score(ep,q):
        aliases=[q['answer']]+q['answer_aliases']
        ep.update(qid=q['id'],f1=max(compute_f1(a,ep['answer']) for a in aliases),em=max(compute_exact(a,ep['answer']) for a in aliases))
        gold={f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']}
        predicted=set(ep['support_ids']);overlap=len(predicted&gold)
        precision=overlap/len(predicted) if predicted else 0;recall=overlap/len(gold)
        ep['support_f1']=2*precision*recall/(precision+recall) if precision+recall else 0
        return ep
    def collect(qid):
        remaining();q=raw['train'][qid];rows,vectors=corpus(q);path=args.output/'teacher'/f'{qid}.json'
        if path.exists():
            obj=json.loads(path.read_text());ep=obj['episode'];rs=obj['records'];fs=np.load(path.with_suffix('.npz'))['features']
        else:
            ep,rs,fs=run_episode(q['question'],rows,vectors,embedder,reader,prompt,'teacher',deadline=deadline)
            for r in rs:r['qid']=qid
            ep=score(ep,q);save(Path('teacher')/path.name,{'episode':ep,'records':rs,'contract':contract})
            np.savez_compressed(path.with_suffix('.npz'),features=np.asarray(fs,dtype=np.float32))
        records.extend(rs);features.extend(fs);episodes.append(ep)
        save('teacher_progress.json',{'completed_questions':len(episodes),'natural_states':len(records),
            'episodes_with_errors':sum(bool(e['errors']) for e in episodes)})
        print(json.dumps({'teacher_qid':qid,'states':len(rs),'f1':ep['f1'],'errors':ep['errors']}),flush=True)
    try:
        phase('compact_loading',contract=contract,teacher_collection_started=False)
        for f in json.loads((repo/'lab/registry/text_integrated_20261016/qwen_model_verified.json').read_text())['files']:
            path=model/f['file']
            if not path.resolve().is_relative_to(Path('/content/local_fast')) or sha256(path)!=f['sha256']:
                raise ValueError('Local model verification failed: '+f['file'])
        embedder=V11ProcessEmbedder();reader=StrictReader(str(model),SimpleNamespace(max_output_tokens=512,context_limit=16384))
        save('embedding_runtime.json',embedder.ready)
        # Common warmup, excluded from timings for every compared condition.
        reader.generate_native([{'role':'user','content':'Return {"answer":"","support_ids":[]}.'}],[],deadline)
        phase('compact_train4_audit',teacher_collection_started=True,teacher_target=4)
        for qid in split['train'][:4]:collect(qid)
        if any(e['errors'] for e in episodes):
            phase('compact_train4_format_nogo',teacher_questions=len(episodes),training_completed=False);backup();return
        phase('compact_native_cost_screen',teacher_questions=4,natural_states=len(records),trained_head=False)
        import torch
        hidden=reader.model.config.hidden_size
        # Biases force the recorded teacher action/slot; this measures actual head
        # compute but cannot establish learned selection or quality.
        reader.checkpoint={'action_weight':torch.zeros((4,hidden),device='cuda'),
            'action_bias':torch.zeros(4,device='cuda'),'slot_weight':torch.zeros((3,5,hidden),device='cuda'),
            'slot_bias':torch.zeros((3,5),device='cuda')}
        cost_rows=[]
        for repeat in range(3):
            for number,r in enumerate(records):
                remaining();q=raw['train'][r['qid']];rows,vectors=corpus(q)
                bucket='FINAL' if r['action']=='FINAL' else ('reuse' if r['slot']>0 else 'generate')
                row={'repeat':repeat,'qid':r['qid'],'loop':r['loop'],'bucket':bucket,'conditions':{}}
                order=['native','integrated','always_generate'];random.Random(20261016+repeat*1000+number).shuffle(order)
                for policy in order:
                    ck=reader.checkpoint
                    ck['action_bias'].zero_();ck['action_bias'][r['action_index']]=1
                    ck['slot_bias'].zero_()
                    if r['slot']>=0:ck['slot_bias'][r['action_index'],r['slot']]=1
                    reader._sync();tick=time.perf_counter()
                    if policy=='native':
                        decision,usage,failure=reader.generate_native(r['messages'],r['history'],deadline)
                        action=decision['action'] if decision else None;value=decision.get('arguments') if decision else None
                        if not failure and action!=r['action']:failure='Native replay action changed'
                        calls=[usage]
                    else:
                        opts=options_for(q['question'],r['history'])
                        choice,_,usage=reader.capture(r['messages'],opts,'head',deadline);calls=[usage];action=choice['action']
                        value=choice['arguments'] if policy=='integrated' else None;failure=None
                        if value is None:
                            value,usage,failure=reader.continue_cached(action,r['history'],deadline);calls.append(usage)
                        else:reader.pending=None
                    if not failure and action!='FINAL':TextCorpus(rows,vectors,embedder).call(action,value)
                    row['conditions'][policy]={'seconds':time.perf_counter()-tick,'failure':failure,'calls':calls,
                        'arguments_exact':action=='FINAL' or (value is not None and normalize_args(action,value)==normalize_args(action,r['teacher_arguments']))}
                cost_rows.append(row);save('cost_rows.json',cost_rows)
                print(json.dumps({'screen_repeat':repeat,'qid':r['qid'],'loop':r['loop'],'bucket':bucket,
                    'seconds':{p:round(v['seconds'],3) for p,v in row['conditions'].items()}}),flush=True)
        screen=cost_summary(cost_rows);save('cost_screen.json',screen)
        if not screen['passed']:
            phase('compact_cost_nogo',speed_gate_passed=False,teacher_questions=4,training_completed=False,
                  reason='Slower, uncertain, missing bucket coverage or invalid path; no teacher32 expansion');backup();return
        phase('compact_teacher32',speed_gate_passed=True,teacher_target=32)
        backup()
        for qid in split['train'][4:32]:collect(qid)
        info,checkpoint=train_heads(records,features,contract)
        if not info['gate']['passed']:
            phase('compact_teacher48_coverage_extension',teacher_target=48,coverage=info['gate'])
            backup()
            for qid in split['train'][32:48]:collect(qid)
            info,checkpoint=train_heads(records,features,contract)
        save('training.json',info)
        if checkpoint is None:
            phase('compact_coverage_nogo',teacher_questions=len(episodes),coverage=info['gate'],training_completed=False);backup();return
        torch.save(checkpoint,args.output/'head.pt')
        if checkpoint['contract']!=contract or tuple(checkpoint['actions'])!=ACTIONS:raise ValueError('Checkpoint contract mismatch')
        reader.checkpoint={k:checkpoint[k].to('cuda') for k in ['action_weight','action_bias','slot_weight','slot_bias']}
        phase('compact_head_evaluation',teacher_questions=len(episodes),training_completed=True)
        backup();results=[]
        conditions=['native','integrated']
        if screen['policies']['always_generate']['passed']:conditions.append('always_generate')
        for number,qid in enumerate(holdout):
            q=raw['dev'][qid];rows,vectors=corpus(q);order=list(conditions);random.Random(20261016+number).shuffle(order)
            for policy in order:
                remaining();rel=Path('evaluation')/policy/(qid+'.json')
                if (args.output/rel).exists():ep=json.loads((args.output/rel).read_text())
                else:
                    ep,_,_=run_episode(q['question'],rows,vectors,embedder,reader,prompt,
                        'native' if policy=='native' else 'head','always_generate' if policy=='always_generate' else 'reuse',deadline)
                    ep=score(ep,q);ep['policy']=policy;save(rel,ep)
                results.append(ep);print(json.dumps({'eval_qid':qid,'policy':policy,'f1':ep['f1'],'errors':ep['errors']}),flush=True)
        summary={p:{'n':len([r for r in results if r['policy']==p]),**{k:float(np.mean([r[k] for r in results if r['policy']==p]))
            for k in ['f1','em','support_f1','seconds','tool_calls','answer_format_valid','prompt_tokens','completion_tokens','reused_argument_calls']},
            'failures':sum(bool(r['errors']) for r in results if r['policy']==p)} for p in conditions}
        native=summary['native'];integrated=summary['integrated']
        save('summary.json',{'policies':summary,'controlled_latency_claim':False,
            'retained':integrated['failures']==0 and integrated['f1']>=native['f1'] and integrated['seconds']<native['seconds'],
            'note':'Small fixed pilot. End-to-end rollouts may have different step counts; matched-state cost screen is separate.'})
        phase('compact_complete',training_completed=True,evaluation_completed=True,teacher_questions=len(episodes));backup()
    except TimeoutError as error:
        phase('compact_budget_limited',error=str(error));backup()
    except Exception as error:
        phase('compact_failed',error=type(error).__name__+': '+str(error));backup();raise
    finally:
        if embedder:embedder.close()


if __name__=='__main__':main()
