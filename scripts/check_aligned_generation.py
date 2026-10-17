"""Frozen24 train state repaired-generation check plus fixed precision probe."""
import argparse,json,time,hashlib,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lsdrag.text_integrated_aligned import AlignedReader,syntax_prefix
from lsdrag.text_integrated import options_for
from lsdrag.text_decision_benchmark import TOOLS,TextCorpus,public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/content/local_fast/lsdrag-text-v1');repo=Path(__file__).resolve().parents[1];parent=root/'results/text_integrated_20261016_v5_fast_only';old=root/'results/improvement_v2_stage1_20261017'
    import fcntl
    lock=(root/'improvement_v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    states=[json.loads(p.read_text()) for p in sorted(old.glob('state_*.json'))];records=[]
    for s in states:
        rs=json.loads((parent/'teacher'/f'{s["qid"]}.json').read_text())['records'];records.append(next(r for r in rs if r['loop']==s['step']))
    manifest={'pid':os.getpid(),'states':[[s['qid'],s['step']] for s in states],'precision_probe_states':[0,4],'numeric_gate':'FP32 cache/full max_abs<=0.001 and equal argmax on two predeclared states; record BF16/full logits separately','functional_gate':'24 valid repaired calls matching current native values with zero extra full prefill','source':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in ['src/lsdrag/text_integrated_aligned.py','scripts/check_aligned_generation.py']},'started_epoch':time.time()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));started=time.perf_counter();deadline=started+900
    model=root/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
    reader=AlignedReader(model,SimpleNamespace(max_output_tokens=512,context_limit=16384));torch=reader.torch
    raw={}
    with (root/'source/musique/musique_ans_v1.0_train.jsonl').open() as f:
        for line in f:
            q=json.loads(line)
            if q['id'] in {s['qid'] for s in states}:raw[q['id']]=q
    embedder=V11ProcessEmbedder();results=[];precision=[]
    try:
        for s,r in zip(states,records):
            native,nu,ne=reader.generate_native(r['messages'],r['history'],deadline)
            reader.capture(r['messages'],options_for(raw[s['qid']]['question'],r['history']),'probe',deadline)
            value,usage,error=reader.continue_cached(s['action'],r['history'],deadline)
            expected=native.get('final',native.get('arguments')) if native else None
            result={'qid':s['qid'],'step':s['step'],'action':s['action'],'native':native,'native_usage':nu,'native_error':ne,'value':value,'usage':usage,'error':error,'matches_native':bool(native and native['action']==s['action'] and value==expected),'matches_historical':value==s['native_value']}
            if value is not None and s['action']!='FINAL':
                corpus=TextCorpus(public_paragraphs(raw[s['qid']]),np.load(parent/'index'/f'{s["qid"]}.npy'),embedder)
                result['result']=corpus.call(s['action'],value)
            results.append(result);(a.output/f'state_{len(results):02d}.json').write_text(json.dumps(result,indent=2));print('state',len(results),'match',result['matches_native'],'error',error,flush=True)
        embedder.close()
        # Two states fixed before this run; full logits saved, not only top5.
        with torch.inference_mode():
            for dtype in [torch.bfloat16,torch.float32]:
                reader.model.to(dtype=dtype)
                for i in [0,4]:
                    if time.perf_counter()>deadline:raise TimeoutError('Probe budget')
                    s=states[i];r=records[i];inputs=reader.prepare(r['messages'],TOOLS);prefix=reader.tokenizer.encode(syntax_prefix(s['action'])[0],add_special_tokens=False);token=torch.tensor([prefix],device=reader.model.device);mask=torch.cat([inputs['attention_mask'],torch.ones_like(token)],-1);ids=torch.cat([inputs['input_ids'],token],-1)
                    initial=reader.model(**inputs,use_cache=True,logits_to_keep=1)
                    warm=reader.model(input_ids=token,attention_mask=mask,past_key_values=initial.past_key_values,use_cache=True,logits_to_keep=1)
                    cold=reader.model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
                    x=warm.logits[0,-1].float();y=cold.logits[0,-1].float();err=float((x-y).abs().max());tag=str(dtype).split('.')[-1]
                    np.savez(a.output/f'logits_{tag}_{i}.npz',cached=x.cpu().numpy(),cold=y.cpu().numpy(),input_ids=ids.cpu().numpy(),attention_mask=mask.cpu().numpy(),expected_position_ids=np.arange(ids.shape[1]))
                    precision.append({'state':i,'dtype':tag,'max_abs_error':err,'argmax_equal':bool(x.argmax()==y.argmax()),'cache_length':int(warm.past_key_values.get_seq_length()),'input_length':ids.shape[1]});print(precision[-1],flush=True)
                    del initial,warm,cold
        functional=all(r['matches_native'] and r['error'] is None and r['usage']['full_prefill_count']==0 for r in results)
        numeric=all(p['max_abs_error']<=.001 and p['argmax_equal'] and p['cache_length']==p['input_length'] for p in precision if p['dtype']=='float32')
        summary={'phase':'aligned_check_complete','pid':None,'states':len(results),'matches_native':sum(r['matches_native'] for r in results),'matches_historical':sum(r['matches_historical'] for r in results),'functional_passed':functional,'precision':precision,'fp32_numeric_passed':numeric,'gate_passed':functional and numeric,'worker_seconds':time.perf_counter()-started,'limitations':'Two-state precision diagnostic and24 training-state functional checks, not heldout policy quality. BF16 original absolute cutoff unchanged; FP32 test diagnoses numerical path separately.'}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:embedder.close()
if __name__=='__main__':main()
