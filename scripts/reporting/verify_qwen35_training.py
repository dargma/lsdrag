"""Raw official-score, token, split and prefix audit for fresh9B training stages."""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import torch
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=ROOT/'results'/args.run
    status=read(run/'status.json');assert status['phase']=='complete';spec=read(run/'spec.json');plan=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json')
    assert set(spec['qids'])<=set(plan['train']) and not set(spec['qids'])&set(plan['development64'])
    if 'source_sha256' in spec:
        for p,h in spec['source_sha256'].items():assert sha(REPO/p)==h,p
    for p,h in spec.get('hashes',{}).items():assert sha(p)==h,p
    raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open())};assert set(spec['qids'])<=raw.keys()
    sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
    episodes={};hashes={};total=0;errors=0;groups={k:dict(n=0,total_generated=0) for k in ['correct','incorrect','partial']}
    for path in sorted((run/'episodes').glob('*.json')):
        e=read(path);hashes[str(path)]=sha(path);assert e['qid'] in spec['qids'];q=raw[e['qid']];answers=[q['answer']]+q['answer_aliases']
        assert abs(e['f1']-max(compute_f1(a,e['answer']) for a in answers))<1e-10
        em=max(compute_exact(a,e['answer']) for a in answers);assert e.get('em',em)==em
        assert e['generated_tokens']==sum(u.get('generated_tokens',0) for u in e['reader_calls'])
        for u in e['reader_calls']:
            if 'token_ids' in u:assert len(u['token_ids'])==u['generated_tokens']
        total+=e['generated_tokens'];errors+=bool(e['errors']);episodes[path.name]=e
        for label in ['correct' if em else 'incorrect']+(['partial'] if not em and 0<e['f1']<1 else []):groups[label]['n']+=1;groups[label]['total_generated']+=e['generated_tokens']
    assert len(episodes)==status['episodes'] and total==status['generated']
    teacher='teacher' in args.run;tensor=run/('teacher_states.pt' if teacher else 'states.pt');records=torch.load(tensor,weights_only=True,map_location='cpu');coverage=defaultdict(set);seen=set()
    if teacher:
        counts=Counter()
        for r in records:
            e=episodes[f"native_teacher_{r['qid']}.json"];decisions=[d for d in e['decisions'] if not d['forced']];d=decisions[counts[r['qid']]];counts[r['qid']]+=1
            assert r['action']==d['action'] and r['legal'].tolist()==d['legal'];coverage[r['action']].add(r['qid'])
        assert sum(len([d for d in e['decisions'] if not d['forced']]) for e in episodes.values())==len(records)
    else:
        for r in records:
            key=(r['qid'],r['turn']);assert key not in seen;seen.add(key)
            parent=episodes[f"{r['qid']}_collector.json"];prefills=[u for u in parent['reader_calls'] if u.get('pack_slots')];d=parent['decisions'][r['turn']]
            assert not d['forced'] and r['prefix_sha256']==prefills[r['turn']]['messages_sha256']
            assert d['legal']==r['legal'].tolist() and d['policy_probs']==r['reference_probs']
            expected=set(torch.where(r['legal'])[0].tolist()) if spec['mode']=='all_actions' else {r['continue_action'],3}
            assert {s['action'] for s in r['samples']}==expected
            for s in r['samples']:
                e=episodes[f"{r['qid']}_t{r['turn']}_a{s['action']}_r0.json"]
                assert e['reader_calls'][0]['messages_sha256']==r['prefix_sha256'] and e['decisions'][0]['action']==s['action']
                assert all(e[a]==s[b] for a,b in [('f1','f1'),('em','em'),('generated_tokens','generated'),('prompt_tokens','inputs'),('errors','errors')])
        assert len(records)==sum(sum(not d['forced'] for d in e['decisions']) for n,e in episodes.items() if n.endswith('_collector.json'))
    assert {r['qid'] for r in records}==set(spec['qids'])
    for r in records:assert r['pack'].shape==(37,4096) and torch.isfinite(r['pack']).all()
    for g in groups.values():g['mean_generated']=g['total_generated']/g['n'] if g['n'] else None
    report=dict(verified=True,run=args.run,scope='Training-source episodes; branch samples are correlated, not independent QA evaluation',questions=len(spec['qids']),episodes=len(episodes),states=len(records),errors=errors,correctness=groups,action_question_coverage={a:len(q) for a,q in coverage.items()},tensor_sha256=sha(tensor),episode_sha256=hashes)
    out=REPO/'lab/reports'/args.run;out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
