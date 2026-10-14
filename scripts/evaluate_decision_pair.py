"""Frozen fresh M3 comparison: each question/arm once, same local HF Reader."""
import argparse
from dataclasses import asdict
import json
import os
import random
import subprocess
import time
from pathlib import Path
import torch
import transformers
from lsdrag.agent import Agent,AgentConfig
from lsdrag.hf_reader import HFReader
from lsdrag.embedding_worker import ProcessEmbedder
from lsdrag.io import read_jsonl,dump_json,sha256
from lsdrag.retrieval import Retrieval,VectorIndex
from lsdrag.metrics import official_scorer,answer_scores


def gpu_pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.work_root;repo=Path(__file__).resolve().parents[1]
    if a.output.exists(): raise ValueError('Refuse repeated evaluation; inspect existing results')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(): raise ValueError('Commit code before final evaluation')
    if gpu_pids(): raise RuntimeError('GPU has active processes; defer timing comparison')
    questions_path=r/'data/manifests/jev_v1.0/fresh_m3_test20/questions.jsonl'
    questions=read_jsonl(questions_path)
    plan=json.loads((repo/'lab/registry/jev_v1.0/label_collection_plan.json').read_text())
    assert {q['qid'] for q in questions}==set(plan['fresh_m3_test_qids']) and len(questions)==20
    assert all(set(q)=={'qid','question'} for q in questions)
    rng=random.Random(20261014);rng.shuffle(questions)
    first=['B-HF']*10+['D-head']*10;rng.shuffle(first)
    schedule=[(q,arm) for q,initial in zip(questions,first) for arm in ([initial,'D-head' if initial=='B-HF' else 'B-HF'])]
    config=AgentConfig();prompt=(repo/'prompts/search_linked.txt').read_text()
    dump_json(a.output/'manifest.json',{'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'schedule':[{'qid':q['qid'],'arm':arm} for q,arm in schedule],'questions_sha256':sha256(questions_path),'head_sha256':sha256(a.head),'config':asdict(config),'torch':torch.__version__,'transformers':transformers.__version__,'prompt_sha256':sha256(repo/'prompts/search_linked.txt'),'timing':'sequential same HF model; load/warmup excluded, per-question preparation/decision/args/tools/fallback/final included'})
    reader=HFReader(str(r/'hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2'),config,a.head)
    trained_head=reader.head
    embedder=ProcessEmbedder(r/'embedding-env/bin/python',r/'hf')
    indexes={c:VectorIndex(r/'indexes_final288'/c,r/'data/ir_final288'/('figures.jsonl' if c.endswith('image') else 'chunks.jsonl')) for c in ['gemma_text','gemma_image']}
    # Short, answer-independent warmup; no holdout history or labels.
    reader.head=None
    old_limit=config.max_output_tokens
    reader.config=AgentConfig(max_output_tokens=16)
    reader.chat([{'role':'user','content':'Reply OK.'}],None)
    reader.config=config
    embedder.encode(['initialization'],'query')
    torch.cuda.reset_peak_memory_stats()
    results=[]
    try:
        for q,arm in schedule:
            allowed_pids={os.getpid(),embedder.process.pid}
            if gpu_pids()-allowed_pids:
                raise RuntimeError('GPU contention appeared; stop before next question')
            reader.head=trained_head if arm=='D-head' else None
            # Fresh caches for both arms; warmup never leaks retrieved test results.
            retrieval=Retrieval(r/'data/ir_final288',indexes,{'gemma':embedder},'G1')
            result=Agent(retrieval,prompt,config,reader).run(q['qid'],q['question'])
            result['decision_arm']=arm
            dump_json(a.output/arm/(q['qid']+'.json'),result)
            if gpu_pids()-allowed_pids:
                dump_json(a.output/'timing_invalid.json',{'qid':q['qid'],'arm':arm,'reason':'GPU contention during question; do not use its timing'})
                raise RuntimeError('GPU contention during question; preserve result, no automatic retry')
            if result['status']=='failed': raise RuntimeError(f'Infrastructure failure {q["qid"]}/{arm}: {result["errors"]}')
            results.append(result)
            print(json.dumps({'completed':len(results),'arm':arm,'qid':q['qid'],'seconds':result['seconds'],'answers':result['answers']}),flush=True)
    finally:
        embedder.close()
    # Gold is loaded only after both arms finish all questions.
    gold={q['qid']:q for q in read_jsonl(r/'data/manifests/jev_v1.0/fresh_m3_test20/gold.jsonl')}
    scorer=official_scorer(r/'upstream/m3docrag');rows=[]
    for result in results:
        rows.append({'qid':result['qid'],'arm':result['decision_arm'],**answer_scores(result,gold[result['qid']],scorer),'seconds':result['seconds'],'reader_calls':result['reader_call_count'],'completion_tokens':result['completion_tokens'],'fallback_count':sum(c['usage'].get('fallback_count',0) for c in result['reader_calls']),'delivered_figures':result['delivered_figure_ids'],'tool_seconds':sum(t['seconds'] for t in result['trajectory']),**{key:sum(c['usage'].get(key,0) for c in result['reader_calls']) for key in ['prefill_seconds','head_seconds','prefix_seconds','decode_seconds']}})
    summary={}
    import statistics
    for arm in ['B-HF','D-head']:
        subset=[x for x in rows if x['arm']==arm]
        summary[arm]={'n':len(subset),**{k:statistics.mean(x[k] for x in subset) for k in ['em','f1','seconds','reader_calls','completion_tokens','fallback_count','tool_seconds','prefill_seconds','head_seconds','prefix_seconds','decode_seconds']},'median_seconds':statistics.median(x['seconds'] for x in subset)}
    dump_json(a.output/'summary.json',{'arms':summary,'rows':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'limitation':'20 paired questions; no strong noninferiority/p95/calibration claim'})

if __name__=='__main__': main()
