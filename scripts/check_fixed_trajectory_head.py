"""Frozen train4 state replay: learned heads dispatch without altering step counts.

Future states always come from the native trace, not student observations. This
is a functional check on training states, not autonomous QA or heldout quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from lsdrag.text_integrated import ACTIONS,StrictReader,options_for
from lsdrag.text_decision_benchmark import TextCorpus,public_paragraphs,canonical
from lsdrag.v11_embedding_worker import V11ProcessEmbedder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    manifest=json.loads((args.run/'manifest.json').read_text())
    repo=Path(__file__).resolve().parents[1]
    for file,digest in manifest['source_sha256'].items():
        if hashlib.sha256((repo/file).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+file)
    checkpoint=torch.load(args.run/'head.pt',map_location='cpu',weights_only=True)
    if tuple(checkpoint['actions'])!=ACTIONS:raise ValueError('Action order mismatch')
    launch=json.loads((args.run/'launch.json').read_text())
    if checkpoint['contract']!=launch['contract']:raise ValueError('Checkpoint contract mismatch')
    qids=manifest['train_qids'][:4];raw={}
    with Path('/content/local_fast/lsdrag-text-v1/source/musique/musique_ans_v1.0_train.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in qids:raw[row['id']]=row
    started=time.perf_counter();deadline=started+300;checks=[];embedder=None
    try:
        embedder=V11ProcessEmbedder()
        reader=StrictReader(manifest['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384))
        reader.checkpoint={k:checkpoint[k].to('cuda') for k in ['action_weight','action_bias','slot_weight','slot_bias']}
        native_counts={};head_counts={}
        for qid in qids:
            records=json.loads((args.run/'teacher'/f'{qid}.json').read_text())['records']
            native_counts[qid]=sum(r['action']!='FINAL' for r in records);head_counts[qid]=0
            corpus=TextCorpus(public_paragraphs(raw[qid]),np.load(args.run/'index'/f'{qid}.npy'),embedder)
            for record in records:
                choice,_,capture=reader.capture(record['messages'],options_for(raw[qid]['question'],record['history']),'head',deadline)
                action=choice['action']
                value,usage,failure=reader.continue_cached(action,record['history'],deadline)
                if not failure and action!='FINAL':
                    result=corpus.call(action,value);head_counts[qid]+=1
                else:result=None
                checks.append({'qid':qid,'step':record['loop'],'native_action':record['action'],'head_action':action,
                    'action_matches':action==record['action'],'failure':failure,'capture':capture,'generation':usage,
                    'result_count':len(result) if result is not None else None,
                    'native_state_sha256':hashlib.sha256(canonical(record['messages']).encode()).hexdigest(),
                    'student_results_used_as_future_input':False})
                report={'kind':'fixed native-state train4 functional replay','checkpoint_sha256':hashlib.sha256((args.run/'head.pt').read_bytes()).hexdigest(),
                    'passed':all(c['failure'] is None and c['action_matches'] for c in checks),
                    'natural_native_states':len(checks),'native_tool_calls':native_counts,'head_tool_calls':head_counts,
                    'same_step_counts_so_far':native_counts==head_counts,'checks':checks,
                    'heldout_quality_or_speed_claim':False,'teacher_forced_observations':True,
                    'worker_seconds':time.perf_counter()-started}
                (args.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
                print(qid,record['loop'],action,failure or 'PASS',flush=True)
        if native_counts!=head_counts or not report['passed']:raise ValueError('Fixed-state functional check failed')
    finally:
        if embedder:embedder.close()


if __name__=='__main__':main()
