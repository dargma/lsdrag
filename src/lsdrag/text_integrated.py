"""Fresh four-action/text argument-reuse pilot. No D3 runtime or old labels."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from jsonschema import validate

from .text_only_reader import TextOnlyReader
from .text_decision_benchmark import TOOLS, FINAL_SCHEMA, TextCorpus, canonical, execution_options

ACTIONS = ('keyword_search', 'semantic_search', 'read_chunk', 'FINAL')
SELECT = ('The runtime can directly execute these complete argument candidates or request new arguments. '
          'Candidates are drawn only from the observed history and may be unsuitable. '
          'Continue the ordinary native tool-call/final-answer protocol; do not output option letters. '
          'If the runtime supplies an assistant prefix, complete that selected operation. '
          'Available execution candidates:\n')


def options_for(question, trajectory):
    options = execution_options(question, trajectory)
    observed = observed_ids(trajectory)
    return [o for o in options if o['action'] != 'read_chunk' or observed]


def observed_ids(trajectory):
    return sorted({r['chunk_id'] for t in trajectory if isinstance(t['result'], list)
                   for r in t['result'] if isinstance(r, dict) and isinstance(r.get('chunk_id'), str)})


def normalize_args(action, value):
    value = copy.deepcopy(value)
    if action.endswith('search'):
        value.setdefault('top_k', 5)
    return value


def slots_for(options, action):
    return [o for o in options if o['action'] == action]


def target_slot(options, action, arguments):
    if action == 'FINAL':
        return -1
    expected = canonical(normalize_args(action, arguments))
    for i, option in enumerate(slots_for(options, action)):
        if option['arguments'] is not None and canonical(normalize_args(action, option['arguments'])) == expected:
            return i
    return 0


def selection_messages(messages, options):
    return copy.deepcopy(messages)+[{'role':'user','content':SELECT+'\n'.join(
        f'{chr(65+i)}: {canonical(o)}' for i,o in enumerate(options))}]


def argument_schema(action, trajectory):
    schema = copy.deepcopy(next(t['function']['parameters'] for t in TOOLS if t['function']['name']==action))
    if action == 'read_chunk':
        ids = observed_ids(trajectory)
        if not ids:
            raise ValueError('read_chunk is illegal before any ID is observed')
        schema['properties']['chunk_ids']['items']['enum'] = ids
    return schema


def teacher_schema(trajectory):
    branches = []
    for action in ACTIONS:
        if action == 'read_chunk' and not observed_ids(trajectory):
            continue
        props = {'action':{'type':'string','enum':[action]}}
        required = ['action']
        if action != 'FINAL':
            props['arguments'] = argument_schema(action, trajectory)
            required.append('arguments')
        branches.append({'type':'object','properties':props,'required':required,'additionalProperties':False})
    return {'oneOf':branches}


def native_schema(trajectory):
    branches=[]
    for action in ACTIONS[:-1]:
        if action=='read_chunk' and not observed_ids(trajectory):continue
        branches.append({'type':'object','properties':{'name':{'type':'string','enum':[action]},
            'arguments':argument_schema(action,trajectory)},'required':['name','arguments'],'additionalProperties':False})
    return {'oneOf':branches}


class StrictReader(TextOnlyReader):
    """Schema mask is applied during decoding; no output unwrapping or repair."""
    def __init__(self, path, config):
        super().__init__(path, config)
        from lmformatenforcer import TokenEnforcerTokenizerData
        # Same tokenizer bridge algorithm as LMFE's MIT-licensed Transformers
        # integration, using its public core API because its old HF import path
        # is incompatible with Transformers5.18. No third-party files patched.
        zero = self.tokenizer.encode('0', add_special_tokens=False)[-1]
        special = set(self.tokenizer.all_special_ids)
        regular = []
        for i in range(len(self.tokenizer)):
            if i not in special:
                after = self.tokenizer.decode([zero,i])[1:]
                plain = self.tokenizer.decode([i])
                regular.append((i, after, len(after)>len(plain)))
        self.grammar_data = TokenEnforcerTokenizerData(regular,
            lambda ids:self.tokenizer.decode(ids).rstrip('�'), self.tokenizer.eos_token_id,
            False, len(self.tokenizer))
        self.checkpoint = None
        self.readout_verified = False

    def generate_schema(self, messages, schema, mode, instruction, deadline=float('inf')):
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        torch = self.torch
        start = time.perf_counter()
        if start >= deadline:
            raise TimeoutError('Worker budget before generation')
        # Request-specific system instructions prevent FINAL instructions from
        # competing with argument-only or teacher decision generation.
        modes = {
            'arguments': 'You generate ONLY the argument JSON object for the tool already selected by the runtime. '
                'Do not answer the original question. Do not output name, function, arguments wrapper, tool-call envelope, '
                'analysis or markdown. Use the original question and observed evidence to form useful arguments.',
            'final': 'Answer the original question using the observed text evidence. Return ONLY final JSON. '
                'The answer must be the shortest correct entity, name, date, number or phrase; no explanation. '
                'Use an empty answer if evidence is insufficient. Cite supporting observed chunk IDs.',
            'teacher': 'Choose the next operation to solve the original multi-hop question from text evidence. '
                'Use keyword_search for short literal entities, semantic_search for conceptual queries, '
                'read_chunk to inspect relevant observed paragraphs, or FINAL when enough evidence is present. '
                'Return exactly one operation JSON with action and, for a tool, its arguments. '
                'Do not use future information, invent IDs, or output analysis or an answer in this decision.'}
        request = copy.deepcopy(messages) if mode=='native' else [{'role':'system','content':modes[mode]}]+[
            copy.deepcopy(m) for m in messages if m['role']!='system']+[
            {'role':'user','content':instruction+'\nRequired JSON Schema: '+canonical(schema)}]
        inputs = self.prepare(request, TOOLS if mode=='native' else None)
        if inputs['input_ids'].shape[1]+self.config.max_output_tokens > self.config.context_limit:
            raise ValueError('Generation context exceeds fixed budget')
        if mode=='native':
            from lmformatenforcer import UnionParser,SequenceParser,StringParser
            parser=UnionParser([SequenceParser([StringParser('<tool_call>\n'),JsonSchemaParser(schema),
                                                StringParser('\n</tool_call>')]),JsonSchemaParser(FINAL_SCHEMA)])
        else:parser=JsonSchemaParser(schema)
        enforcer = TokenEnforcer(self.grammar_data, parser)
        self._sync(); preprocess = time.perf_counter()-start
        tick = time.perf_counter()
        generated, constrained = [], 0
        sequence = inputs['input_ids'][0].tolist()
        mask = inputs['attention_mask']
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=True, logits_to_keep=1)
            self._sync(); prefill = time.perf_counter()-tick; tick = time.perf_counter()
            for _ in range(self.config.max_output_tokens):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Worker budget during generation')
                allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
                if not allowed:
                    raise RuntimeError('JSON grammar has no valid continuation')
                logits = output.logits[0,-1].float()
                token_id = allowed[int(logits[allowed].argmax().item())]
                constrained += int(token_id != int(logits.argmax().item()))
                generated.append(token_id); sequence.append(token_id)
                if token_id in self.stop_ids:
                    break
                token = torch.tensor([[token_id]],device=self.model.device)
                mask = torch.cat([mask,torch.ones_like(token)],-1)
                output = self.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,
                                    use_cache=True,logits_to_keep=1)
        self._sync()
        text = self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        usage = {'prompt_tokens':int(inputs['input_ids'].shape[1]),'completion_tokens':len(generated),
                 'generated_tokens':len(generated),'preprocess_seconds':preprocess,'prefill_seconds':prefill,
                 'decode_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-start,
                 'grammar_changed_tokens':constrained,'raw_json':text,'mode':mode,
                 'request_sha256':hashlib.sha256(canonical(request).encode()).hexdigest(),
                 'full_prefill_count':1,'cache_reused':False,
                 'finish_reason':'stop' if generated and generated[-1] in self.stop_ids else 'length'}
        try:
            if usage['finish_reason']!='stop':
                raise ValueError('truncated_generation')
            if mode=='native':
                if text.startswith('<tool_call>'):
                    raw=text[len('<tool_call>'):].strip()
                    if not raw.endswith('</tool_call>'):raise ValueError('Incomplete native tool call')
                    value=json.loads(raw[:-len('</tool_call>')].strip());validate(value,schema)
                    value={'action':value['name'],'arguments':value['arguments']}
                else:
                    value=json.loads(text);validate(value,FINAL_SCHEMA)
                    value={'action':'FINAL','final':value}
            else:
                value = json.loads(text); validate(value,schema)
            if any(t in text for t in ['<think>','<|channel>thought','<|think|>']):
                raise ValueError('Unexpected thinking output')
            return value, usage, None
        except Exception as error:
            return None, usage, type(error).__name__+': '+str(error)

    def capture(self, messages, options, mode, deadline):
        torch = self.torch
        start = time.perf_counter()
        if start >= deadline:
            raise TimeoutError('Worker budget before selection')
        request = selection_messages(messages,options)
        inputs = self.prepare(request,TOOLS)
        if inputs['input_ids'].shape[1]+1 > self.config.context_limit:
            raise ValueError('Selection context exceeds fixed budget')
        self._sync(); tick = time.perf_counter()
        with torch.inference_mode():
            if mode == 'first_token':
                captured=[]
                hook=self.model.model.norm.register_forward_hook(lambda mod,args,out:captured.append(out[:,-1].detach().clone()))
                try:
                    output=self.model(**inputs,use_cache=True,logits_to_keep=1)
                finally:hook.remove()
                feature=captured[0][0].float().cpu()
                logits=output.logits[0,-1,self.label_ids[:len(options)]].float()
                choice=options[int(logits.argmax().item())]
            else:
                output=self.model.model(**inputs,use_cache=True,return_dict=True)
                feature=output.last_hidden_state[0,-1].float().cpu()
                if mode=='head':
                    ck=self.checkpoint
                    if ck is None:raise ValueError('No trained checkpoint loaded')
                    feature_gpu=feature.to(self.model.device)
                    scores=ck['action_weight']@feature_gpu+ck['action_bias']
                    legal=torch.tensor([any(o['action']==a for o in options) for a in ACTIONS],device=scores.device)
                    action_index=int(scores.masked_fill(~legal,-torch.inf).argmax().item())
                    action=ACTIONS[action_index]; slots=slots_for(options,action)
                    if action=='FINAL':choice=slots[0]
                    else:
                        logits=ck['slot_weight'][action_index]@feature_gpu+ck['slot_bias'][action_index]
                        choice=slots[int(logits[:len(slots)].argmax().item())]
                else:choice=None
            self.pending={'past':output.past_key_values,'mask':inputs['attention_mask'],
                          'ids':inputs['input_ids'][0].tolist()}
            if mode=='teacher' and not self.readout_verified:
                direct=feature.clone(); captured=[]
                hook=self.model.model.norm.register_forward_hook(lambda mod,args,out:captured.append(out[:,-1].detach().clone()))
                try:self.model(**inputs,use_cache=False,logits_to_keep=1)
                finally:hook.remove()
                torch.testing.assert_close(direct,captured[0][0].float().cpu(),rtol=1e-4,atol=1e-4)
                self.readout_verified=True
                extra=int(inputs['input_ids'].shape[1])
            else:extra=0
        self._sync()
        return choice, feature.numpy().copy(), {
            'prompt_tokens':int(inputs['input_ids'].shape[1])+extra,'completion_tokens':0,
            'prefill_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-start,
            'mode':mode,'vocab_projection':mode=='first_token','readout_extra_check_tokens':extra,
            'full_prefill_count':1+int(extra>0),'cache_retained':True,
            'messages_sha256':hashlib.sha256(canonical(request).encode()).hexdigest()}

    def continue_cached(self, action, trajectory, deadline=float('inf')):
        """One batched forced prefix + cached decode, never a second full prefill."""
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer,SequenceParser,StringParser
        torch=self.torch; start=time.perf_counter()
        pending=getattr(self,'pending',None);self.pending=None
        if pending is None:raise RuntimeError('Missing/consumed decision KV cache')
        if start>=deadline:raise TimeoutError('Budget before cache continuation')
        schema=FINAL_SCHEMA if action=='FINAL' else argument_schema(action,trajectory)
        key='answer' if action=='FINAL' else {'keyword_search':'keywords','semantic_search':'query','read_chunk':'chunk_ids'}[action]
        argprefix='{"'+key+'":'
        prefix=argprefix if action=='FINAL' else '<tool_call>\n{"name":'+json.dumps(action)+',"arguments":'+argprefix
        parser=JsonSchemaParser(schema)
        for char in argprefix:parser=parser.add_character(char)
        suffix='' if action=='FINAL' else '}\n</tool_call>'
        if suffix:parser=SequenceParser([parser,StringParser(suffix)])
        enforcer=TokenEnforcer(self.grammar_data,parser)
        ids=self.tokenizer.encode(prefix,add_special_tokens=False)
        token=torch.tensor([ids],device=self.model.device)
        mask=torch.cat([pending['mask'],torch.ones_like(token)],-1)
        if mask.shape[1]+self.config.max_output_tokens>self.config.context_limit:
            raise ValueError('Cached continuation exceeds fixed context')
        sequence=pending['ids']+ids;generated=[];constrained=0
        self._sync();tick=time.perf_counter()
        with torch.inference_mode():
            output=self.model(input_ids=token,attention_mask=mask,past_key_values=pending['past'],
                              use_cache=True,logits_to_keep=1)
            self._sync();prefix_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            for _ in range(self.config.max_output_tokens-len(ids)):
                if time.perf_counter()>=deadline:raise TimeoutError('Budget during cached decode')
                allowed=enforcer.get_allowed_tokens(sequence).allowed_tokens
                if not allowed:raise RuntimeError('No cached grammar continuation')
                logits=output.logits[0,-1].float();tid=allowed[int(logits[allowed].argmax().item())]
                constrained+=int(tid!=int(logits.argmax().item()))
                generated.append(tid);sequence.append(tid)
                if tid in self.stop_ids:break
                token=torch.tensor([[tid]],device=self.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
                output=self.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,
                                  use_cache=True,logits_to_keep=1)
        self._sync();tail=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        text=argprefix+tail
        if suffix and text.endswith(suffix):text=text[:-len(suffix)].rstrip()
        usage={'prompt_tokens':len(ids),'completion_tokens':len(generated),'generated_tokens':len(generated),
            'forced_prefix_tokens':len(ids),'prefix_seconds':prefix_seconds,'decode_seconds':time.perf_counter()-tick,
            'seconds':time.perf_counter()-start,'full_prefill_count':0,'cache_reused':True,
            'grammar_changed_tokens':constrained,'raw_json':text,
            'finish_reason':'stop' if generated and generated[-1] in self.stop_ids else 'length'}
        try:
            if usage['finish_reason']!='stop':raise ValueError('truncated_cached_generation')
            value=json.loads(text);validate(value,schema)
            return value,usage,None
        except Exception as error:return None,usage,type(error).__name__+': '+str(error)


def run_episode(question, rows, vectors, embedder, reader, prompt, selector,
                argument_mode='reuse', deadline=float('inf')):
    start=time.perf_counter(); corpus=TextCorpus(rows,vectors,embedder)
    messages=[{'role':'system','content':prompt},{'role':'user','content':question}]
    trajectory,calls,records,features,errors=[],[],[],[],[]
    final={'answer':'','support_ids':[]}; valid=False; forced=False
    for turn in range(9):
        if time.perf_counter()>=deadline:raise TimeoutError('Worker budget before next decision')
        options=options_for(question,trajectory); forced=turn==8
        # No hidden truncation: a conservative context estimate triggers the
        # same explicit forced FINAL on every policy, and is never a target.
        if not forced:
            length=int(reader.prepare(selection_messages(messages,options),None)['input_ids'].shape[1])
            forced=length+reader.config.max_output_tokens+256>reader.config.context_limit
        args=None; selected=None; feature=None; native_final=None
        if forced:action='FINAL'
        else:
            if selector!='native':
                choice,feature,usage=reader.capture(messages,options,selector,deadline)
                calls.append({'kind':'student_capture' if selector=='teacher' else 'decision','usage':usage})
            if selector in {'teacher','native'}:
                decision,usage,failure=reader.generate_schema(messages,native_schema(trajectory),'native','',deadline)
                calls.append({'kind':'teacher_decision' if selector=='teacher' else 'native_combined','usage':usage})
                if failure:errors.append('teacher_format: '+failure);break
                action=decision['action'];args=decision.get('arguments')
                native_final=decision.get('final')
                if selector=='teacher':
                    slot=target_slot(options,action,args)
                    record={'loop':turn,'action':action,'action_index':ACTIONS.index(action),'slot':slot,
                        'messages':copy.deepcopy(messages),'history':copy.deepcopy(trajectory),
                        'teacher_arguments':copy.deepcopy(args),
                        'options':copy.deepcopy(options),'options_sha256':hashlib.sha256(canonical(options).encode()).hexdigest(),
                        'feature_origin':'student_before_teacher_output','natural':True,
                        'messages_sha256':calls[-2]['usage']['messages_sha256']}
                    records.append(record);features.append(feature)
                reader.pending=None
            else:
                action=choice['action'];selected=choice
                if argument_mode=='reuse':args=choice['arguments']
        if action=='FINAL':
            if native_final is not None:final=native_final;failure=None
            elif not forced:
                final,usage,failure=reader.continue_cached('FINAL',trajectory,deadline)
                calls.append({'kind':'final','usage':usage})
            else:
                final,usage,failure=reader.generate_schema(messages,FINAL_SCHEMA,'final',
                    'Return the shortest evidence-supported answer and supporting chunk IDs as final JSON.',deadline)
                calls.append({'kind':'final','usage':usage})
            if failure:errors.append('final_format: '+failure);final={'answer':'','support_ids':[]}
            else:valid=True
            break
        reused=bool(selector not in {'teacher','native'} and args is not None)
        if args is None:
            args,usage,failure=reader.continue_cached(action,trajectory,deadline)
            calls.append({'kind':'arguments','usage':usage})
            if failure:errors.append('argument_format: '+failure);break
        else:reader.pending=None
        validate(args,argument_schema(action,trajectory))
        tick=time.perf_counter();result=corpus.call(action,args)
        trajectory.append({'tool_name':action,'arguments':args,'result':result,'argument_reused':reused,
                           'seconds':time.perf_counter()-tick})
        tc={'id':f'call_{turn}','type':'function','function':{'name':action,'arguments':canonical(args)}}
        messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]},
                         {'role':'tool','tool_call_id':tc['id'],'content':canonical(result)}])
    episode={'answer':final['answer'],'support_ids':final['support_ids'],'answer_format_valid':valid,
             'errors':errors,'trajectory':trajectory,'reader_calls':calls,'forced_final':forced,
             'tool_calls':len(trajectory),'reused_argument_calls':sum(t['argument_reused'] for t in trajectory),
             'argument_tokens':sum(c['usage'].get('generated_tokens',0) for c in calls if c['kind']=='arguments'),
             'argument_generation_calls':sum(c['kind']=='arguments' for c in calls),
             'prompt_tokens':sum(c['usage'].get('prompt_tokens',0) for c in calls),
             'completion_tokens':sum(c['usage'].get('completion_tokens',0) for c in calls),
             'seconds':time.perf_counter()-start,'selector':selector,'argument_mode':argument_mode}
    return episode,records,features


def train_heads(records, features, contract):
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261016)
    coverage={a:len({r['qid'] for r in records if r['action']==a}) for a in ACTIONS}
    reuse=len({r['qid'] for r in records if r['slot']>0})
    generated=len({r['qid'] for r in records if r['slot']==0})
    qids={r['qid'] for r in records}
    gate={'passed':len(qids)>=30 and min(coverage.values())>=5 and reuse>=5 and generated>=5,
          'natural_questions':len(qids),'per_action_questions':coverage,'reuse_questions':reuse,'generate_questions':generated}
    if not gate['passed']:return {'gate':gate,'trained':False},None
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or len(x)!=len(records) or not np.isfinite(x).all():raise ValueError('Invalid training features')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    tx=torch.from_numpy((x-mean)/scale);y=torch.tensor([r['action_index'] for r in records])
    sy=torch.tensor([r['slot'] for r in records]);nonfinal=sy>=0
    legal=torch.tensor([[any(o['action']==a for o in r['options']) for a in ACTIONS] for r in records])
    slotmask=torch.tensor([[j<len(slots_for(r['options'],r['action'])) for j in range(5)] for r in records])
    action=torch.nn.Linear(x.shape[1],4);slot=torch.nn.Linear(x.shape[1],15)
    optimizer=torch.optim.AdamW(list(action.parameters())+list(slot.parameters()),lr=.001,weight_decay=.01)
    curve=[]
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits=action(tx).masked_fill(~legal,-torch.inf)
        sl=slot(tx).reshape(-1,3,5)[nonfinal,y[nonfinal]].masked_fill(~slotmask[nonfinal],-torch.inf)
        loss=torch.nn.functional.cross_entropy(logits,y)+torch.nn.functional.cross_entropy(sl,sy[nonfinal])
        loss.backward();optimizer.step();curve.append(float(loss.detach()))
    def fold(layer):
        w=layer.weight.detach()/torch.from_numpy(scale)[None,:]
        b=layer.bias.detach()-w@torch.from_numpy(mean)
        torch.testing.assert_close(layer(tx),torch.from_numpy(x)@w.T+b,atol=1e-4,rtol=1e-4)
        assert torch.equal(layer(tx).argmax(-1),(torch.from_numpy(x)@w.T+b).argmax(-1))
        return w,b
    aw,ab=fold(action);sw,sb=fold(slot)
    ck={'format':'text-integrated-head-v1','contract':contract,'actions':ACTIONS,'slot_order':'GENERATE_ARGS, candidates1..4',
        'action_weight':aw,'action_bias':ab,'slot_weight':sw.reshape(3,5,-1),'slot_bias':sb.reshape(3,5),
        'seed':20261016,'epochs':100,'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),
        'train_qids':sorted(qids)}
    with torch.no_grad():
        accuracy=float((action(tx).masked_fill(~legal,-torch.inf).argmax(-1)==y).float().mean())
    return {'gate':gate,'trained':True,'loss':curve,'train_action_accuracy':accuracy,
            'note':'Training imitation diagnostic only, not heldout or QA quality.'},ck
