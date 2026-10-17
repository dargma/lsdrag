"""Read-only CPU audit of frozen stage1 artifacts; never loads a backbone."""
import hashlib,json,os
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from transformers import AutoTokenizer
from lsdrag.text_decision_benchmark import TextCorpus,public_paragraphs,canonical

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/improvement_v2_stage1_20261017'
PARENT=ROOT/'results/text_integrated_20261016_v5_fast_only'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
manifest=json.loads((RUN/'manifest.json').read_text())
split=json.loads((REPO/'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
assert manifest['split']==split
states=[json.loads(p.read_text()) for p in sorted(RUN.glob('state_*.json'))]
assert len(states)==24 and len(manifest['state_keys'])==24
raw={}
with (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open() as stream:
    for line in stream:
        q=json.loads(line)
        if q['id'] in split['diagnostic']:raw[q['id']]=q
model=ROOT/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
class NoEncoder:
    def encode(self,*args,**kwargs):raise RuntimeError('CPU audit forbids new semantic inference')
counts=Counter();byaction={};cases=[];audit=[]
for i,s in enumerate(states):
    qid=s['qid'];assert qid in split['train'] and qid not in split['validation']
    rs=json.loads((PARENT/'teacher'/f'{qid}.json').read_text())['records']
    record=next(r for r in rs if r['loop']==s['step'])
    assert hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()==manifest['state_hashes'][i]
    rendered=tokenizer.apply_chat_template(record['messages'],tools=__import__('lsdrag.text_decision_benchmark',fromlist=['TOOLS']).TOOLS,add_generation_prompt=True,tokenize=False,enable_thinking=False)
    assert tokenizer(rendered)['input_ids']==s['input_token_ids']
    assert s['native_token_ids']==s['prefix_token_ids']+s['exact_off']['token_ids']
    assert tokenizer.decode(s['native_token_ids'],skip_special_tokens=True).strip()==s['native_text']
    assert s['native_value']==s['exact_on']['value']
    assert s['prefix_token_ids']==s['native_token_ids'][:len(s['prefix_token_ids'])]
    p=s['parity'];assert p['cache_length']==p['mask_length']==p['input_length']==len(s['input_token_ids'])+len(s['prefix_token_ids'])
    counts['verified_input_and_state_identity']+=1
    counts['same_boundary_argmax']+=p['cached_argmax']==p['cold_argmax']
    counts['old_absolute_cutoff_failures']+=p['max_abs_error']>0.25
    counts['margin_exceeds_twice_saved_max_error']+=min(p['cached_margin'],p['cold_margin'])>2*p['max_abs_error']
    counts['compact_value_changes']+=not s['compact_matches_native']
    action=s['action'];byaction.setdefault(action,Counter());byaction[action]['states']+=1;byaction[action]['compact_value_changes']+=not s['compact_matches_native']
    # Old oracle flag means crossing the opening brace, not semantic leakage.
    prefix=s['prefix_text'];tail=prefix.split('"arguments": ',1)[1] if action!='FINAL' else ''
    syntax_only=tail in ['', '{', '{"']
    counts['syntax_only_prefixes']+=syntax_only
    counts['overconservative_oracle_flags']+=bool(s['prefix_oracle'] and syntax_only)
    corpus=TextCorpus(public_paragraphs(raw[qid]),np.zeros((len(raw[qid]['paragraphs']),768)),NoEncoder())
    if action in ['keyword_search','read_chunk']:
        for name,args in [('native',s['native_value']),('exact_off',s['exact_off']['value']),('exact_on',s['exact_on']['value']),('compact_on',s['compact_on']['value'])]:
            result=corpus.call(action,args);saved=s['dispatch'][name]
            assert hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()==saved['result_sha256']
            assert sorted(r['chunk_id'] for r in result)==saved['result_ids']
            counts['cpu_handler_replays']+=1
    if action!='FINAL':
        n=s['dispatch']['native'];c=s['dispatch']['compact_on']
        byaction[action]['native_empty']+=n['empty'];byaction[action]['compact_empty']+=c['empty']
        counts['compact_tool_result_changes']+=n['result_sha256']!=c['result_sha256']
        counts['compact_support_recall_losses']+=c['support_recall']<n['support_recall']
    if not s['compact_matches_native']:
        cases.append({'qid':qid,'step':s['step'],'action':action,'native':s['native_value'],'compact':s['compact_on']['value'],'dispatch':s['dispatch']})
    audit.append({'qid':qid,'step':s['step'],'action':action,'max_abs_error':p['max_abs_error'],'minimum_top1_margin':min(p['cached_margin'],p['cold_margin']),'old_cutoff_passed':p['explained'],'boundary_top1_robust_to_recorded_error':min(p['cached_margin'],p['cold_margin'])>2*p['max_abs_error'],'prefix_suffix':tail,'syntax_only':syntax_only})
report={'kind':'CPU-only audit of previously saved GPU outputs; no new model inference','counts':dict(counts),'by_action':{k:dict(v) for k,v in byaction.items()},'cases':cases,'state_audit':audit,
    'input_hashes':{p.name:sha(p) for p in sorted(RUN.glob('*.json'))},
    'findings':[
        'All24 exact-prefix OFF full generated token streams equal native; exact-prefix ON parsed values equal native. Input/state identities and lengths reproduced.',
        '14 states exceed original absolute logit cutoff0.25. All24 boundary top1 decisions agree and margins exceed2*recorded max error; original numeric gate is not retroactively changed.',
        'Differences in BF16 batching/kernel arithmetic are plausible but not proven by saved scalars/top5. Full logits, explicit position tensors and high-precision reference are absent; no definitive cache-bug exclusion.',
        'Compact serialization/prefix path changes7/24 values;6 keyword and1FINAL. Three keyword searches lose support recall; two change nonempty results to empty. This is fixed-state generation-path evidence, not isolated learned-head causality or autonomous QA.',
        'All16 native argument-prefix oracle flags are conservative false positives for semantic payload: saved prefixes end at an opening quote only. Native-derived syntax remains retrospective and not a prevalidated deployment protocol.',
        'Exact-prefix grammar ON matches all24 native values; does not establish grammar irrelevance in all contexts. Compact path also changes forced field prefix, tokenization, prefill/readout and grammar implementation; factors are not fully separated.'
    ],
    'decision':{'cpu_review_complete':True,'stage1_gate_passed':False,'stage2_started':False,'gpu_used_this_review':False,'deployment_fix_validated':False,
        'next':'Prepare deterministic native-style syntax-only continuation and FINAL alignment; freeze before same train-state functional check. Verify cache numerics with targeted full-logit/high-precision evidence if needed; no GPU launch authorized by this CPU-only request.'}}
out=REPO/'lab/reports/text_integrated_20261016/improvement_v3_stage1_cpu_review.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':report['counts'],'by_action':report['by_action'],'decision':report['decision']},ensure_ascii=False,indent=2))
