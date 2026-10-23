"""Bounded head-only policy experiments. Historical reader/profile stays unchanged."""
import copy
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from jsonschema import validate

from .text_integrated import ACTIONS, options_for, argument_schema, advance_json_parser, observed_ids
from .text_integrated_aligned import AlignedReader, syntax_prefix
from .text_decision_benchmark import TOOLS, FINAL_SCHEMA, TextCorpus, canonical

# Slots 0..4: last, question-end, latest-observation-end, last16 mean,
# latest-observation mean; 5..36 are 32 ordered, disjoint context-bin means.
FEATURE_SLOTS = 37


def matched_span(tokenizer, ids, text, last=False):
    """Locate exact content offsets in the rendered sequence, never approximate silently."""
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    start = decoded.rfind(text) if last else decoded.find(text)
    if start < 0:
        raise ValueError('Content span missing from rendered input')
    encoded = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)
    if encoded['input_ids'] != ids:
        raise ValueError('Tokenizer roundtrip differs; span inference unsafe')
    end = start + len(text)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and a < end]
    if not positions:
        raise ValueError('Empty content span')
    return positions[0], positions[-1] + 1


def pack_hidden(hidden, question_span, observation_span):
    """All features are functions of current public prefix; no teacher suffix."""
    n = hidden.shape[0]
    q0, q1 = question_span
    o0, o1 = observation_span
    assert 0 <= q0 < q1 <= n and 0 <= o0 < o1 <= n and n >= 32
    h = hidden.float()
    points = torch.linspace(0, n, 33).long().tolist()
    values = [h[-1], h[q1-1], h[o1-1], h[-16:].mean(0), h[o0:o1].mean(0)]
    values += [h[a:b].mean(0) for a, b in zip(points, points[1:])]
    return torch.stack(values)


class PolicyHead(nn.Module):
    def __init__(self, dim=2560, architecture='mlp', feature=0):
        super().__init__()
        self.architecture, self.feature, self.dim = architecture, feature, dim
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        if architecture == 'attention':
            self.keys = nn.Linear(dim, 64, bias=False)
            self.values = nn.Linear(dim, 64, bias=False)
            self.queries = nn.Parameter(torch.randn(4, 64) * .02)
            self.positions = nn.Parameter(torch.zeros(32, 64))
            self.action = nn.Linear(64, 1)
            width = 64
        elif architecture == 'mlp':
            self.trunk = nn.Sequential(nn.Linear(dim, 128), nn.Tanh())
            self.action = nn.Linear(128, 4)
            self.skip = nn.Linear(dim, 4)
            width = 128
        elif architecture == 'linear':
            self.action = nn.Linear(dim, 4)
            width = dim
        else:
            raise ValueError(architecture)
        self.value = nn.Linear(width, 1)
        self.success = nn.Linear(width, 4)
        self.cost = nn.Linear(width, 4)

    def forward(self, pack, legal):
        x = (pack.float() - self.mean) / self.scale
        if self.architecture == 'attention':
            tokens = x[:, 5:]
            keys = self.keys(tokens) + self.positions
            weights = torch.einsum('ad,btd->bat', self.queries, keys).div(8).softmax(-1)
            context = weights @ self.values(tokens)
            logits = self.action(context).squeeze(-1)
            z = context.mean(1)
        else:
            z = x[:, self.feature]
            if self.architecture == 'mlp':
                residual = self.skip(z)
                z = self.trunk(z)
                logits = self.action(z) + residual
            else:
                logits = self.action(z)
        logits = logits.masked_fill(~legal, -1e9)
        return logits, self.value(z).squeeze(-1), self.success(z), self.cost(z)


def distribution(logits, legal, epsilon=0.):
    p = logits.masked_fill(~legal, -1e9).softmax(-1)
    p = (1-epsilon)*p + epsilon*legal.float()/legal.sum(-1, keepdim=True)
    return torch.distributions.Categorical(probs=p)


def returns_for_episode(steps, quality, cost_weight=.005):
    """Discount=1 episodic returns; each real decision incurs exactly its own cost."""
    result = []
    total = float(quality)
    for step in reversed(steps):
        total -= cost_weight*step['cost_units']
        result.append(total)
    return list(reversed(result))


def clipped_policy_loss(new_logp, old_logp, advantage, clip=.2):
    ratio = (new_logp-old_logp).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()


class PolicyReader(AlignedReader):
    def capture_pack(self, messages, history, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('policy worker deadline')
        start = time.perf_counter()
        inputs = self.prepare(messages, TOOLS)
        ids = inputs['input_ids'][0].tolist()
        if len(ids)+self.config.max_output_tokens+64 > self.config.context_limit:
            raise ValueError('Context limit: no silent truncation')
        question = next(m['content'] for m in messages if m['role']=='user')
        observation = next((m['content'] for m in reversed(messages) if m['role']=='tool'), question)
        qs = matched_span(self.tokenizer, ids, question)
        obs = matched_span(self.tokenizer, ids, observation, last=bool(history))
        self._sync(); tick = time.perf_counter()
        with torch.inference_mode():
            out = self.model.model(**inputs, use_cache=True, return_dict=True)
            pack = pack_hidden(out.last_hidden_state[0], qs, obs)
        self.pending = {'past':out.past_key_values, 'ids':ids, 'mask':inputs['attention_mask'],
                        'last_hidden':out.last_hidden_state[:,-1].detach().clone()}
        self._sync()
        usage = {'prompt_tokens':len(ids), 'completion_tokens':0, 'generated_tokens':0,
                 'full_prefill_count':1, 'prefill_seconds':time.perf_counter()-tick,
                 'seconds':time.perf_counter()-start, 'question_span':qs, 'observation_span':obs,
                 'pack_slots':FEATURE_SLOTS, 'messages_sha256':hashlib.sha256(canonical(messages).encode()).hexdigest()}
        return pack, usage

    def fast_native(self, history, deadline):
        """Native name logits, shared syntax batched, same cache continues payload.

        A legal prefix trie includes FINAL's {"answer": start. This constrained
        baseline differs from unconstrained native, and all prefix work is charged.
        """
        legal = [a for a in ACTIONS if a!='read_chunk' or observed_ids(history)]
        prefixes = {a: ('{"answer":', '{"answer":') if a=='FINAL' else syntax_prefix(a) for a in legal}
        sequences = {a:self.tokenizer.encode(v[0], add_special_tokens=False) for a,v in prefixes.items()}
        pending = self.pending
        emitted = []
        calls = 0
        forced = 0
        tick = time.perf_counter()
        def feed(tokens):
            nonlocal calls
            t = torch.tensor([tokens],device=self.model.device)
            pending['mask'] = torch.cat([pending['mask'],torch.ones_like(t)],-1)
            with torch.inference_mode():
                out = self.model.model(input_ids=t,attention_mask=pending['mask'],past_key_values=pending['past'],use_cache=True,return_dict=True)
            pending['past'] = out.past_key_values
            pending['last_hidden'] = out.last_hidden_state[:,-1].detach().clone()
            pending['ids'].extend(tokens); emitted.extend(tokens); calls += 1
        while True:
            if time.perf_counter()>=deadline: raise TimeoutError('native-fast deadline')
            active = {a:s for a,s in sequences.items() if s[:len(emitted)]==emitted}
            done = [a for a,s in active.items() if len(s)==len(emitted)]
            if done:
                assert len(done)==1
                action=done[0];break
            remaining=list(active.values());common=[]
            for j in range(len(emitted), min(map(len,remaining))):
                if len({s[j] for s in remaining})!=1:break
                common.append(remaining[0][j])
            if common:
                forced+=len(common);feed(common)
            else:
                choices=sorted({s[len(emitted)] for s in remaining})
                with torch.inference_mode():
                    logits=self.model.lm_head(pending['last_hidden'])[0].float()
                tid=choices[int(logits[choices].argmax())];feed([tid])
        self._sync()
        usage={'prompt_tokens':len(emitted), 'generated_tokens':len(emitted)-forced,
               'completion_tokens':len(emitted)-forced,'forced_prefix_tokens':forced,
               'prefix_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-tick,
               'full_prefill_count':0,'prefix_forward_calls':calls,'cache_reused':True,
               'raw_prefix':self.tokenizer.decode(emitted),'mode':'native_trie_batched_syntax'}
        value, tail, error = self.payload_from_current(action, history, prefixes[action][1], deadline, self.config.max_output_tokens-len(emitted))
        return action,value,usage,tail,error

    def payload_from_current(self, action, history, initial_text, deadline, token_budget):
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer
        schema=FINAL_SCHEMA if action=='FINAL' else argument_schema(action,history)
        initial=advance_json_parser(JsonSchemaParser(schema,self.json_config),initial_text)
        parser=initial;pending=self.pending;self.pending=None
        seq=pending['ids'];mask=pending['mask'];past=pending['past'];generated=[];decoded='';complete=False
        tick=time.perf_counter()
        with torch.inference_mode():
            logits=self.model.lm_head(pending['last_hidden'])[0].float()
            for _ in range(token_budget):
                if time.perf_counter()>=deadline:raise TimeoutError('payload deadline')
                tid=int(logits.argmax());proposed=self.tokenizer.decode(generated+[tid],skip_special_tokens=True)
                candidate=advance_json_parser(initial,proposed)
                if tid in self.stop_ids or candidate is None or len(proposed)<=len(decoded):
                    allowed=TokenEnforcer(self.grammar_data,parser).get_allowed_tokens(seq).allowed_tokens
                    tid=allowed[int(logits[allowed].argmax())]
                    proposed=self.tokenizer.decode(generated+[tid],skip_special_tokens=True);candidate=advance_json_parser(initial,proposed)
                    if candidate is None:raise ValueError('native-fast grammar mismatch')
                generated.append(tid);seq.append(tid);decoded=proposed;parser=candidate
                try:json.loads(initial_text+decoded);complete=True;break
                except json.JSONDecodeError:pass
                token=torch.tensor([[tid]],device=self.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
                out=self.model(input_ids=token,attention_mask=mask,past_key_values=past,use_cache=True,logits_to_keep=1)
                past=out.past_key_values;logits=out.logits[0,-1].float()
        self._sync()
        usage={'prompt_tokens':0,'generated_tokens':len(generated),'completion_tokens':len(generated),
               'full_prefill_count':0,'decode_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-tick,
               'cache_reused':True,'raw_json':initial_text+decoded}
        try:
            if not complete:raise ValueError('truncated native-fast payload')
            val=json.loads(initial_text+decoded);validate(val,schema);return val,usage,None
        except Exception as e:return None,usage,str(e)


def rollout(reader, question, rows, vectors, embedder, prompt, head, deadline,
            stochastic=False, epsilon=.05, initial=None, first_action=None, historical=None,
            native_fast=False):
    """head chooses every suffix step; branch rollouts never enter PPO's on-policy batch."""
    corpus=TextCorpus(rows,vectors,embedder)
    messages=copy.deepcopy(initial['messages']) if initial else [{'role':'system','content':prompt},{'role':'user','content':question}]
    history=copy.deepcopy(initial['history']) if initial else []
    initial_calls=len(history);steps=[];usage=[];errors=[];final={'answer':'','support_ids':[]};snapshots=[]
    start=time.perf_counter();forced=False
    while True:
        forced=len(history)>=8
        pack, cu=reader.capture_pack(messages,history,deadline);usage.append(cu)
        legal=torch.tensor([[any(o['action']==a for o in options_for(question,history)) for a in ACTIONS]],device=pack.device)
        if forced:legal[:]=False;legal[:,3]=True
        with torch.no_grad():
            if historical is not None:
                logits=(historical['action_weight']@pack[0]+historical['action_bias']).unsqueeze(0)
                val=torch.zeros(1,device=pack.device)
            elif head is not None:
                logits,val,_,_=head(pack.unsqueeze(0),legal)
            else:logits=torch.zeros((1,4),device=pack.device);val=torch.zeros(1,device=pack.device)
            dist=distribution(logits,legal,epsilon if stochastic else 0.)
            action=int(dist.sample()) if stochastic else int(dist.probs.argmax())
            if first_action is not None and not steps:
                action=first_action;assert bool(legal[0,action])
            logp=float(dist.log_prob(torch.tensor([action],device=pack.device)))
        if len(snapshots)<2 and not forced:
            snapshots.append({'messages':copy.deepcopy(messages),'history':copy.deepcopy(history)})
        name=ACTIONS[action]
        if native_fast and not forced:
            name,payload,pu,tu,error=reader.fast_native(history,deadline);usage.extend([pu,tu]);action=ACTIONS.index(name)
            increment=[cu,pu,tu]
        else:
            payload,tu,error=reader.continue_cached(name,history,deadline);usage.append(tu);increment=[cu,tu]
        generated=sum(u.get('generated_tokens',0) for u in increment)
        prefix=sum(u.get('prompt_tokens',0) for u in increment)
        # Fixed proxy, not wall time: calls + decode/100 + all prefix tokens/10000.
        cost=(name!='FINAL') + generated/100 + prefix/10000
        steps.append({'pack':pack.detach().cpu().to(torch.bfloat16),'legal':legal[0].cpu(),'action':action,
                      'old_logp':logp,'value':float(val[0]),'cost_units':cost,'policy_probs':dist.probs[0].cpu().tolist(),
                      'forced':forced,'branch_intervention':first_action is not None and len(steps)==0})
        if error:
            errors.append(error);break
        if name=='FINAL':final=payload;break
        validate(payload,argument_schema(name,history))
        tick=time.perf_counter();result=corpus.call(name,payload)
        history.append({'tool_name':name,'arguments':payload,'result':result,'argument_reused':False,'seconds':time.perf_counter()-tick})
        tc={'id':f'call_{len(history)-1}','type':'function','function':{'name':name,'arguments':canonical(payload)}}
        messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]}, {'role':'tool','tool_call_id':tc['id'],'content':canonical(result)}])
    reader.pending=None
    episode={'answer':final['answer'],'support_ids':final['support_ids'],'errors':errors,'forced_final':forced,
             'trajectory':history,'tool_calls':len(history)-initial_calls,'initial_tool_calls':initial_calls,
             'seconds':time.perf_counter()-start,'reader_calls':usage,
             'generated_tokens':sum(u.get('generated_tokens',0) for u in usage),
             'prompt_tokens':sum(u.get('prompt_tokens',0) for u in usage),
             'forced_prefix_tokens':sum(u.get('forced_prefix_tokens',0) for u in usage),
             'cost_units':sum(s['cost_units'] for s in steps),
             'decisions':[{k:v for k,v in s.items() if k not in ('pack','legal')}|{'legal':s['legal'].tolist()} for s in steps]}
    return episode,steps,snapshots
