"""Audit online behavior probabilities, reward groups and frozen-parameter invariants."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from lsdrag.decision_policy import distribution
from lsdrag.decision_state_learning import StateHead

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/decision_selected_head_grpo192_20261024_zero_gate'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    m,status,summary=[read(RUN/name) for name in ['manifest.json','status.json','summary.json']]
    assert status['phase']=='complete'
    expected=len(m['train'])*m['repeats']*m['iterations']
    eps=[read(p) for p in (RUN/'episodes').glob('*.json')]
    assert len(eps)==expected==status['episodes']==m['budget_episodes']
    lookup={(e['policy'],e['qid']):e for e in eps}
    assert len(lookup)==expected and {e['qid'] for e in eps}==set(m['train'])
    assert not set(m['train']) & set(m['validation']+m['test'])
    for filename,digest in m['source_sha256'].items():assert sha(REPO/filename)==digest
    assert sha(Path(m['reference']))==m['input_sha256'][m['reference']]
    initial=torch.load(m['reference'],weights_only=True,map_location='cpu')
    for key,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:
        assert sum(e[field] for e in eps)==status[key]
        for e in eps:assert e[field]==sum(c.get(field,0) for c in e['reader_calls'])
    decision_count=0
    for e in eps:
        assert np.isclose(sum(d['cost_units'] for d in e['decisions']),e['cost_units'])
        for d in e['decisions']:
            p=np.asarray(d['policy_probs']);legal=np.asarray(d['legal'])
            assert np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and (p[~legal]==0).all()
            assert not d['branch_intervention'] and legal[d['action']]
            assert abs(float(np.log(p[d['action']]))-d['old_logp'])<2e-5
            decision_count+=1
    rounds=[]
    for iteration in range(m['iterations']):
        before=Path(m['reference']) if iteration==0 else RUN/f'grpo_iteration{iteration-1}.pt'
        old=StateHead(inputs='multi',body='mlp');old.load_state_dict(torch.load(before,weights_only=True,map_location='cpu'));old.eval()
        batch=torch.load(RUN/f'batch_{iteration}.pt',weights_only=True,map_location='cpu')
        assert all(row['pack'].dtype==torch.float32 for row in batch)
        groups={}
        for row in batch:groups.setdefault((row['qid'],row['rep']),[]).append(row)
        assert len(groups)==len(m['train'])*m['repeats']
        for (qid,rep),rows in groups.items():
            ep=lookup[f'train_i{iteration}_r{rep}',qid]
            decisions=[d for d in ep['decisions'] if not d['forced']]
            assert len(rows)==len(decisions)
            for row,d in zip(rows,decisions):
                assert not row['forced'] and not row['branch_intervention']
                assert row['action']==d['action'] and row['legal'].tolist()==d['legal']
                assert row['episode_steps']==len(rows)
                assert np.isclose(row['episode_reward'],ep['f1']-.03*ep['cost_units'])
                assert row['episode_f1']==ep['f1']
                assert np.isclose(row['old_logp'],d['old_logp'])
        saved_adv=read(RUN/f'advantages_{iteration}.json')
        for q in m['train']:
            es=[lookup[f'train_i{iteration}_r{rep}',q] for rep in range(m['repeats'])]
            r=np.array([e['f1']-.03*e['cost_units'] for e in es])
            expected_adv=np.zeros_like(r) if max(e['f1'] for e in es)==0 else (r-r.mean())/(r.std()+1e-6)
            assert np.allclose(saved_adv[q],expected_adv,atol=1e-12)
        max_error=0.
        with torch.no_grad():
            for first in range(0,len(batch),32):
                rows=batch[first:first+32];x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);action=torch.tensor([r['action'] for r in rows])
                replay=distribution(old(x,legal)[0],legal).log_prob(action)
                saved=torch.tensor([r['old_logp'] for r in rows])
                error=float((replay-saved).abs().max());max_error=max(max_error,error)
                assert torch.allclose(replay,saved,atol=1e-3,rtol=1e-3),error
        after=torch.load(RUN/f'grpo_iteration{iteration}.pt',weights_only=True,map_location='cpu')
        changed=[]
        for name,value in after.items():
            assert torch.isfinite(value).all()
            if not name.startswith('net.'):assert torch.equal(value,initial[name]),name
            elif not torch.equal(value,initial[name]):changed.append(name)
        reward_spreads=[np.ptp([groups[q,rep][0]['episode_reward'] for rep in range(m['repeats'])]) for q in m['train']]
        rounds.append(dict(iteration=iteration,states=len(batch),groups=len(groups)//m['repeats'],groups_with_nonzero_reward_spread=sum(float(v)>1e-8 for v in reward_spreads),max_cpu_replay_logprob_error=max_error,changed_actor_parameters=changed,optimizer_steps=4*((len(batch)+31)//32),checkpoint_sha256=sha(RUN/f'grpo_iteration{iteration}.pt')))
    assert summary['checkpoint_sha256']==rounds[-1]['checkpoint_sha256']
    result=dict(audit_passed=True,episodes=len(eps),decisions=decision_count,rounds=rounds,errors=sum(bool(e['errors']) for e in eps),forced_final=sum(e['forced_final'] for e in eps),counts={k:status[k] for k in ['generated','inputs']},note='Training audit only. CPU/GPU replay tolerance1e-3; worker checked GPU replay at2e-4 before each update. No heldout quality inference.')
    out=REPO/'lab/reports'/RUN.name;out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
