"""Improvement v2 stage 1: frozen train states, exact token/cache diagnostics.

No validation/heldout reads, teacher expansion, or historical runtime mutations.
"""
import argparse, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
from jsonschema import validate
from lsdrag.text_integrated import StrictReader, argument_schema, native_schema, options_for, advance_json_parser
from lsdrag.text_decision_benchmark import TOOLS, FINAL_SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def native_value(text, history):
    if text.startswith('<tool_call>') and text.endswith('</tool_call>'):
        value=json.loads(text[len('<tool_call>'):-len('</tool_call>')].strip())
        validate(value,native_schema(history));return value['name'],value['arguments']
    value=json.loads(text);validate(value,FINAL_SCHEMA);return 'FINAL',value


def greedy(reader, output, mask, deadline, limit=512, trace=None):
    torch=reader.torch;ids=[]
    for _ in range(limit):
        if time.perf_counter()>deadline:raise TimeoutError('Stage1 wall budget')
        logits=output.logits[0,-1].float();tid=int(logits.argmax());ids.append(tid)
        if trace is not None:
            values,indices=logits.topk(5);trace.append({'ids':indices.tolist(),'logits':values.tolist(),'margin':float(values[0]-values[1])})
        if tid in reader.stop_ids:break
        token=torch.tensor([[tid]],device=reader.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
        output=reader.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
    return ids


def prefix_boundary(reader, ids, action):
    text=reader.tokenizer.decode(ids,skip_special_tokens=True)
    if action=='FINAL':return [],'',False
    # End after the arguments opening brace. Exact native token boundary may
    # include semantic characters: flag that diagnostic prefix as oracle.
    import re
    match=re.search(r'"arguments"\s*:\s*\{',text)
    if not match:raise ValueError('No native argument boundary')
    end=match.end()
    for n in range(1,len(ids)+1):
        prefix=reader.tokenizer.decode(ids[:n],skip_special_tokens=True)
        if len(prefix)>=end:
            return ids[:n],prefix[match.end()-1:],len(prefix)>end
    raise ValueError('Missing token boundary')


def start_cached(reader, inputs, prefix):
    torch=reader.torch
    output=reader.model(**inputs,use_cache=True,logits_to_keep=1)
    mask=inputs['attention_mask']
    if prefix:
        token=torch.tensor([prefix],device=reader.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
        output=reader.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
    return output,mask


def constrained(reader, inputs, prefix, argprefix, schema, deadline):
    from lmformatenforcer import JsonSchemaParser,TokenEnforcer
    torch=reader.torch;parser=advance_json_parser(JsonSchemaParser(schema,reader.json_config),argprefix)
    if parser is None:raise ValueError('Native prefix violates schema')
    initial=parser;decoded='';generated=[];changes=0
    output,mask=start_cached(reader,inputs,prefix)
    sequence=inputs['input_ids'][0].tolist()+prefix
    for _ in range(512-len(prefix)):
        if time.perf_counter()>deadline:raise TimeoutError('Stage1 wall budget')
        logits=output.logits[0,-1].float();tid=int(logits.argmax());candidate_text=reader.tokenizer.decode(generated+[tid],skip_special_tokens=True)
        candidate=advance_json_parser(initial,candidate_text)
        if tid in reader.stop_ids or candidate is None or len(candidate_text)<=len(decoded):
            allowed=TokenEnforcer(reader.grammar_data,parser).get_allowed_tokens(sequence).allowed_tokens
            chosen=allowed[int(logits[allowed].argmax())];changes+=int(chosen!=tid);tid=chosen
            candidate_text=reader.tokenizer.decode(generated+[tid],skip_special_tokens=True)
            candidate=advance_json_parser(initial,candidate_text)
            if candidate is None:raise ValueError('Grammar/token mismatch')
        generated.append(tid);sequence.append(tid);parser=candidate;decoded=candidate_text
        try:
            value=json.loads(argprefix+decoded);validate(value,schema)
            return {'value':value,'token_ids':generated,'grammar_changed_tokens':changes,'valid':True,'forced_prefix_tokens':len(prefix)}
        except json.JSONDecodeError:pass
        if tid in reader.stop_ids:break
        token=torch.tensor([[tid]],device=reader.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
        output=reader.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
    return {'value':None,'token_ids':generated,'valid':False,'raw':argprefix+decoded,'grammar_changed_tokens':changes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--minutes',type=float,default=15);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path('/content/local_fast/lsdrag-text-v1');parent=root/'results/text_integrated_20261016_v5_fast_only'
    a.output.mkdir(parents=True,exist_ok=False)
    import fcntl,os
    lock=(root/'improvement_v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    split=json.loads((repo/'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
    assert len(split['train'])==24 and len(split['validation'])==8 and not set(split['train'])&set(split['validation'])
    records=[]
    for qid in split['diagnostic']:
        rs=json.loads((parent/'teacher'/f'{qid}.json').read_text())['records']
        # Before new outcomes: first, first read/intermediate, final; at most3/q.
        selected=[rs[0]]
        middle=next((r for r in rs[1:-1] if r['action']=='read_chunk'),next(iter(rs[1:-1]),None))
        if middle:selected.append(middle)
        if rs[-1] not in selected:selected.append(rs[-1])
        records.extend(selected)
    assert len(records)<=24 and {r['qid'] for r in records}<=set(split['train'])
    manifest={'split':split,'state_keys':[[r['qid'],r['loop']] for r in records],
        'state_hashes':[digest(r) for r in records],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stage':'generation_path_diagnostic','pid':os.getpid(),'worker_minutes':a.minutes}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Require the actual current host's observed continuous window before load.
    import sys
    sys.path.insert(0,str(repo/'scripts'));from watch_gpu_schedule import sample
    idle=json.loads(Path('/content/local_fast/lsdrag-recovery/20261017/idle_status.json').read_text())
    if not idle.get('eligible') or time.time()-idle['epoch']>65 or not sample(idle['gpu_uuid'] if 'gpu_uuid' in idle else 'GPU-619f1d71-c0b4-21cf-abcd-4a50a8549f3f')['idle']:
        raise RuntimeError('15-minute idle gate not currently satisfied')
    started=time.perf_counter();deadline=started+a.minutes*60
    model=root/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
    reader=StrictReader(model,SimpleNamespace(max_output_tokens=512,context_limit=16384));torch=reader.torch
    results=[]
    with torch.inference_mode():
        for record in records:
            if time.perf_counter()>deadline:raise TimeoutError('Stage1 wall budget')
            inputs=reader.prepare(record['messages'],TOOLS)
            output=reader.model(**inputs,use_cache=True,logits_to_keep=1)
            native_trace=[];native_ids=greedy(reader,output,inputs['attention_mask'],deadline,trace=native_trace);del output
            text=reader.tokenizer.decode(native_ids,skip_special_tokens=True).strip()
            result={'qid':record['qid'],'step':record['loop'],'input_token_ids':inputs['input_ids'][0].tolist(),'native_token_ids':native_ids,'native_text':text}
            try:
                action,value=native_value(text,record['history']);prefix,argprefix,oracle=prefix_boundary(reader,native_ids,action)
                schema=FINAL_SCHEMA if action=='FINAL' else argument_schema(action,record['history'])
                cached,mask=start_cached(reader,inputs,prefix)
                fullids=torch.cat([inputs['input_ids'],torch.tensor([prefix],device=reader.model.device,dtype=torch.long)],-1)
                cold=reader.model(input_ids=fullids,attention_mask=mask,use_cache=False,logits_to_keep=1)
                x=cached.logits[0,-1].float();y=cold.logits[0,-1].float();err=float((x-y).abs().max());cx=int(x.argmax());cy=int(y.argmax())
                margin=lambda z:float(z.topk(2).values.diff().abs()[0])
                parity={'max_abs_error':err,'cached_argmax':cx,'cold_argmax':cy,'cached_margin':margin(x),'cold_margin':margin(y),
                    'cache_length':int(cached.past_key_values.get_seq_length()),'mask_length':mask.shape[1],'input_length':fullids.shape[1],
                    'cached_top5':x.topk(5).indices.tolist(),'cold_top5':y.topk(5).indices.tolist()}
                parity['explained']=bool(parity['cache_length']==fullids.shape[1] and (cx==cy or max(margin(x),margin(y))<=2*err) and err<=0.25)
                del cold
                offtrace=[];off=greedy(reader,cached,mask,deadline,512-len(prefix),trace=offtrace);del cached
                divergence=next((i for i,(x,y) in enumerate(zip(native_ids[len(prefix):],off)) if x!=y),None)
                divergence_info=None if divergence is None else {'index':divergence+len(prefix),'native':native_trace[divergence+len(prefix)],'cached':offtrace[divergence]}
                parity['continuation_explained']=divergence is None or max(divergence_info['native']['margin'],divergence_info['cached']['margin'])<=0.5
                offtext=reader.tokenizer.decode(prefix+off,skip_special_tokens=True).strip()
                try:offaction,offvalue=native_value(offtext,record['history']);offerror=None
                except Exception as e:offaction=offvalue=None;offerror=str(e)
                exact=constrained(reader,inputs,prefix,argprefix,schema,deadline)
                reader.capture(record['messages'],options_for(record['messages'][1]['content'],record['history']),'probe',deadline)
                compact,usage,failure=reader.continue_cached(action,record['history'],deadline)
                result.update(action=action,native_value=value,prefix_token_ids=prefix,prefix_oracle=oracle,prefix_text=reader.tokenizer.decode(prefix),parity=parity,
                    exact_off={'token_ids':off,'value':offvalue,'action':offaction,'error':offerror,'tokens_equal_native':prefix+off==native_ids,'first_divergence':divergence_info},exact_on=exact,
                    compact_on={'value':compact,'usage':usage,'error':failure},exact_on_matches_native=exact['value']==value,compact_matches_native=compact==value)
            except Exception as e:result['error']=type(e).__name__+': '+str(e)
            results.append(result)
            (a.output/f'state_{len(results):02d}.json').write_text(json.dumps(result,indent=2))
            summary={'completed_states':len(results),'planned_states':len(records),'worker_seconds':time.perf_counter()-started,
                'all_parity_explained':all(r.get('parity',{}).get('explained',False) and r.get('parity',{}).get('continuation_explained',False) for r in results),
                'errors':sum('error' in r for r in results),'phase':'stage1_running','pid':os.getpid(),
                'exact_off_token_equal':sum(r.get('exact_off',{}).get('tokens_equal_native',False) for r in results),
                'exact_on_match':sum(r.get('exact_on_matches_native',False) for r in results),
                'compact_match':sum(r.get('compact_matches_native',False) for r in results)}
            (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    summary['phase']='stage1_complete';summary['pid']=None
    summary['next']='Review token divergence and schema failures before stage2; no automatic quality claim.'
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
