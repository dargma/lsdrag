"""Native-style syntax continuation for improvement v3; historical v5 unchanged."""
import json,time
from types import SimpleNamespace
from jsonschema import validate
from .text_integrated import StrictReader,argument_schema,advance_json_parser
from .text_decision_benchmark import FINAL_SCHEMA


def syntax_prefix(action):
    if action=='FINAL':return '', ''
    if action not in ('keyword_search','semantic_search','read_chunk'):raise ValueError(action)
    return '<tool_call>\n{"name": "'+action+'", "arguments": {"', '{"'


class AlignedReader(StrictReader):
    def capture(self,*args,**kwargs):
        last=[]
        hook=self.model.model.norm.register_forward_hook(lambda m,a,o:last.append(o[:,-1].detach().clone()))
        try:result=super().capture(*args,**kwargs)
        finally:hook.remove()
        self.pending['last_hidden']=last[0]
        return result

    def continue_cached(self,action,trajectory,deadline=float('inf')):
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer
        torch=self.torch;start=time.perf_counter();pending=self.pending;self.pending=None
        if pending is None:raise RuntimeError('Missing decision cache')
        prefix,argprefix=syntax_prefix(action);ids=self.tokenizer.encode(prefix,add_special_tokens=False)
        schema=FINAL_SCHEMA if action=='FINAL' else argument_schema(action,trajectory)
        parser=advance_json_parser(JsonSchemaParser(schema,self.json_config),argprefix);initial=parser
        generated=[];decoded='';changed=0;complete=False;sequence=pending['ids']+ids
        mask=pending['mask'];full=len(sequence)
        if full+self.config.max_output_tokens>self.config.context_limit:raise ValueError('Context limit')
        self._sync();tick=time.perf_counter()
        with torch.inference_mode():
            if ids:
                token=torch.tensor([ids],device=self.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
                output=self.model(input_ids=token,attention_mask=mask,past_key_values=pending['past'],use_cache=True,logits_to_keep=1)
            else:
                output=SimpleNamespace(logits=self.model.lm_head(pending['last_hidden']).unsqueeze(1),past_key_values=pending['past'])
            self._sync();prefix_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            for _ in range(self.config.max_output_tokens-len(ids)):
                if time.perf_counter()>=deadline:raise TimeoutError('Worker deadline')
                logits=output.logits[0,-1].float();tid=int(logits.argmax())
                proposed=self.tokenizer.decode(generated+[tid],skip_special_tokens=True)
                candidate=advance_json_parser(initial,proposed)
                if tid in self.stop_ids or candidate is None or len(proposed)<=len(decoded):
                    allowed=TokenEnforcer(self.grammar_data,parser).get_allowed_tokens(sequence).allowed_tokens
                    chosen=allowed[int(logits[allowed].argmax())];changed+=int(tid!=chosen);tid=chosen
                    proposed=self.tokenizer.decode(generated+[tid],skip_special_tokens=True);candidate=advance_json_parser(initial,proposed)
                    if candidate is None:raise ValueError('Grammar token mismatch')
                generated.append(tid);sequence.append(tid);decoded=proposed;parser=candidate
                try:json.loads(argprefix+decoded);complete=True;break
                except json.JSONDecodeError:pass
                if tid in self.stop_ids:break
                token=torch.tensor([[tid]],device=self.model.device);mask=torch.cat([mask,torch.ones_like(token)],-1)
                output=self.model(input_ids=token,attention_mask=mask,past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
        self._sync();text=argprefix+decoded
        usage={'prompt_tokens':len(ids),'completion_tokens':len(generated),'generated_tokens':len(generated),'forced_prefix_tokens':len(ids),'prefix_seconds':prefix_seconds,'decode_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-start,'full_prefill_count':0,'cache_reused':True,'grammar_changed_tokens':changed,'raw_json':text,'token_ids':generated,'prefix_token_ids':ids,'finish_reason':'schema_complete' if complete else 'length','syntax_profile':'native-spaced-v3','final_uses_unforced_cached_logits':action=='FINAL'}
        try:
            if not complete:raise ValueError('Truncated generation')
            value=json.loads(text);validate(value,schema);return value,usage,None
        except Exception as e:return None,usage,type(e).__name__+': '+str(e)
