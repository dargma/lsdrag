"""Train-only hypothetical reference encoding accounting; no inference changes."""
import collections,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path('/content/local_fast/lsdrag-text-v1')
MODEL=ROOT/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
RUN=ROOT/'results/decision_current_policy_refresh48_20261024'
ACTIONS=['keyword_search','semantic_search','read_chunk','FINAL']

def main():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    totals=collections.Counter();by={a:collections.Counter() for a in ACTIONS};sources={}
    for path in sorted((RUN/'episodes').glob('*_collector.json')):
        raw=path.read_bytes();sources[path.name]=hashlib.sha256(raw).hexdigest();ep=json.loads(raw)
        aliases={}
        for i,d in enumerate(ep['decisions']):
            # Only IDs already observed before this decision can acquire aliases.
            if i>0 and i-1<len(ep['trajectory']):
                result=ep['trajectory'][i-1]['result']
                if isinstance(result,list):
                    for row in result:
                        if isinstance(row,dict) and isinstance(row.get('chunk_id'),str):
                            key=row['chunk_id']
                            if key not in aliases:aliases[key]=f'd{len(aliases)}'
            call=ep['reader_calls'][2*i+1];text=call['raw_json'];name=ACTIONS[d['action']]
            actual=call['generated_tokens'];by[name]['actual_generated']+=actual
            payload=json.loads(text);known=[];unknown=[]
            for key in ['chunk_ids','support_ids']:
                for doc in payload.get(key,[]):
                    (known if doc in aliases else unknown).append(doc)
            # Replacement only in quoted reference fields; search/answer text unchanged.
            altered=dict(payload)
            for key in ['chunk_ids','support_ids']:
                if key in altered:altered[key]=[aliases.get(doc,doc) for doc in altered[key]]
            original=json.dumps(payload,ensure_ascii=False)
            shorter=json.dumps(altered,ensure_ascii=False)
            delta=len(tok.encode(original,add_special_tokens=False))-len(tok.encode(shorter,add_special_tokens=False))
            assert delta>=0
            by[name]['reference_occurrences']+=len(known)
            by[name]['unobserved_references']+=len(unknown)
            by[name]['hypothetical_payload_token_delta']+=delta
            by[name]['calls']+=1
        totals['questions']+=1;totals['actual_generated']+=ep['generated_tokens']
    assert totals['questions']==48 and sum(x['actual_generated'] for x in by.values())==totals['actual_generated']
    savings=sum(x['hypothetical_payload_token_delta'] for x in by.values())
    report=dict(scope='Train48 stochastic collector episodes only',totals=dict(totals),actions={k:dict(v) for k,v in by.items()},hypothetical_delta=savings,hypothetical_fraction=savings/totals['actual_generated'],source_sha256=sources,limitations=['JSON re-encoding delta is an accounting estimate, not measured decoder token savings.','Aliases use prior observed public IDs only, no gold support data.','Prompt, choices, final answers and trajectories may change under a new interface.','Both baseline and head policies need matched-interface evaluation; encoding savings are engineering, not learned-policy evidence.'])
    out=Path('lab/reports/decision_current_policy_refresh48_20261024/reference_serialization.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
