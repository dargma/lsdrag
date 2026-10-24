"""Qwen3.5 text-only reader under the frozen A-RAG JSON tool protocol.

The common Qwen3 JSON tool template is deliberate: Qwen3.5's official tool
serialization is XML. Both 9B baseline and heads use this same adapter, so
baseline must be described as common-protocol, not unmodified native Qwen3.5.
"""
import copy
from types import SimpleNamespace
from .decision_policy_recovery import RecoveryPolicyReader
from .text_integrated import grammar_config

class Qwen35Reader(RecoveryPolicyReader):
    common_template_path = '/content/local_fast/lsdrag-code/lab/registry/qwen35_9b_h3_20261024/common_chat_template.jinja'

    def __init__(self,path,config):
        import torch
        from transformers import Qwen3_5ForCausalLM,AutoTokenizer
        from lmformatenforcer import TokenEnforcerTokenizerData
        from pathlib import Path
        self.torch,self.config=torch,config
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.processor=SimpleNamespace(tokenizer=self.tokenizer)
        self.common_template=Path(self.common_template_path).read_text()
        self.model,loading=Qwen3_5ForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise RuntimeError(f'Incomplete language backbone: {loading}')
        self.loading_info=loading
        self.model.eval().requires_grad_(False)
        stop=self.model.generation_config.eos_token_id
        self.stop_ids=set(stop if isinstance(stop,list) else [stop])
        self.labels=list('ABCDEFGHIJKLMNOP');self.label_ids=[self.tokenizer.encode(x,add_special_tokens=False)[0] for x in self.labels]
        zero=self.tokenizer.encode('0',add_special_tokens=False)[-1];special=set(self.tokenizer.all_special_ids);regular=[]
        for i in range(len(self.tokenizer)):
            if i not in special:
                after=self.tokenizer.decode([zero,i])[1:];plain=self.tokenizer.decode([i]);regular.append((i,after,len(after)>len(plain)))
        self.grammar_data=TokenEnforcerTokenizerData(regular,lambda ids:self.tokenizer.decode(ids).rstrip('�'),self.tokenizer.eos_token_id,False,len(self.tokenizer))
        self.json_config=grammar_config(regular);self.checkpoint=None;self.readout_verified=False

    def prepare(self,messages,tools):
        if any(not isinstance(m.get('content'),(str,type(None))) for m in messages):raise ValueError('Text only')
        rendered=self.tokenizer.apply_chat_template(copy.deepcopy(messages),tools=tools or None,chat_template=self.common_template,add_generation_prompt=True,tokenize=False,enable_thinking=False)
        # Official Qwen3.5 enable_thinking=False suffix: closed empty block.
        # These are INPUT tokens, not hidden generated reasoning or free output.
        rendered+='<think>\n\n</think>\n\n'
        return self.tokenizer(rendered,return_tensors='pt').to(self.model.device)
