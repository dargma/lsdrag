"""Training collector coverage of existing public-context argument candidates."""
import collections
import hashlib
import json
from pathlib import Path
from lsdrag.text_integrated import options_for, target_slot, observed_ids

ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_current_policy_refresh48_20261024')
OUT=Path('lab/reports/decision_current_policy_refresh48_20261024/read_reuse_coverage.json')

def main():
    counts=collections.Counter()
    lengths=collections.Counter()
    records=[]
    for path in sorted((ROOT/'episodes').glob('*_collector.json')):
        ep=json.loads(path.read_text())
        history=[]
        for turn,step in enumerate(ep['trajectory']):
            if step['tool_name']=='read_chunk':
                # Question affects semantic candidates only, never read candidates.
                opts=options_for('',history)
                args=step['arguments']
                selected=args['chunk_ids']
                assert set(selected)<=set(observed_ids(history))
                slot=target_slot(opts,'read_chunk',args)
                tokens=ep['reader_calls'][turn*2+1]['generated_tokens']
                assert ep['decisions'][turn]['action']==2
                counts['read_calls']+=1
                counts['read_generated']+=tokens
                counts['existing_candidate_exact_matches']+=slot>0
                counts['exact_match_generated']+=tokens if slot>0 else 0
                lengths[len(selected)]+=1
                records.append({'qid':ep['qid'],'turn':turn,'observed_ids':len(observed_ids(history)),
                    'selected_count':len(selected),'matching_slot':slot,'generated':tokens,
                    'episode_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            history.append(step)
    result={'counts':dict(counts),'selected_id_count_histogram':dict(lengths),'records':records,
            'limitations':['Train48 stochastic collectors only; no evaluation.',
             'Exact coverage is an oracle upper bound, not learned selector accuracy or measured savings.',
             'Candidate list uses existing observed public results, without support labels.',
             'Changing ordering/subsets can change downstream context; retain generation fallback.']}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
