"""Verify native hard-label data, costs, train split and common initialization."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    run = ROOT/'results/decision_shared_teacher192_20261023'
    status, spec, summary = [read(run/p) for p in ['status.json','spec.json','summary.json']]
    assert status['phase']=='complete' and spec['split']=='train'
    split = read(ROOT/'results/decision_state_policy_iteration_20261023/manifest.json')
    assert spec['qids']==split['train']
    assert not set(spec['qids']) & set(split['validation']+split['test'])
    for filename, digest in spec['source_sha256'].items():
        assert sha(REPO/filename)==digest
    rows = torch.load(run/'teacher_states.pt',weights_only=True)
    assert sha(run/'teacher_states.pt')==summary['teacher_states_sha256']
    assert len(rows)==summary['teacher_states']
    episodes = [read(p) for p in (run/'episodes').glob('*.json')]
    assert len(episodes)==len(spec['qids'])==status['episodes']
    assert {e['qid'] for e in episodes}==set(spec['qids'])
    for e in episodes:
        selected = [r for r in rows if r['qid']==e['qid']]
        decisions = [d for d in e['decisions'] if not d['forced']]
        assert len(selected)==len(decisions)
        for r,d in zip(selected,decisions):
            assert set(r)=={'pack','legal','action','qid','episode_f1'}
            assert r['pack'].shape==(37,2560) and torch.isfinite(r['pack']).all()
            assert r['action']==d['action'] and r['legal'].tolist()==d['legal']
            assert bool(r['legal'][r['action']]) and r['episode_f1']==e['f1']
            assert d['policy_probs'] is None and d['old_logp'] is None
        for metric in ['generated_tokens','prompt_tokens']:
            assert e[metric]==sum(c.get(metric,0) for c in e['reader_calls'])
    for metric,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:
        assert sum(e[metric] for e in episodes)==status[key]
    init = ROOT/'results/decision_shared_initialization_20261023'
    m=read(init/'manifest.json')
    assert m['train']==spec['qids'] and m['teacher_states_sha256']==summary['teacher_states_sha256']
    assert sha(init/'head.pt')==m['sha256']
    weights=torch.load(init/'head.pt',weights_only=True,map_location='cpu')
    x=torch.stack([r['pack'][0] for r in rows]).float()
    assert torch.allclose(weights['mean'],x.mean(0))
    assert torch.allclose(weights['scale'],x.std(0,unbiased=False).clamp_min(.1))
    assert all(torch.isfinite(t).all() for t in weights.values())
    result=dict(audit_passed=True,questions=len(episodes),states=len(rows),errors=sum(bool(e['errors']) for e in episodes),forced_final=sum(e['forced_final'] for e in episodes),state_sha256=summary['teacher_states_sha256'],head_sha256=m['sha256'],train_only_normalization_verified=True,native_action_alignment_verified=True,costs={k:status[k] for k in ['generated','inputs']},note='Data/implementation audit; does not establish policy quality or target attainment.')
    out=REPO/'lab/reports/decision_shared_teacher192_20261023';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
