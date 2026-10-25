"""Validate proposed gated bonus and rescore existing paths; no policy training."""
import json,itertools,math
from pathlib import Path
from lsdrag.decision_correct_short_reward import correct_short_reward
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/qwen35_grpo_token100_20261025')
REPO=Path(__file__).resolve().parents[2]
def main():
    for t in [0,1,16,100,122,256,1000,1000000]:
        for f1 in [0.,.25,.5,1.]:
            wrong=correct_short_reward(0,f1,t,True)
            assert wrong['correct_short_bonus']==0
            assert wrong['reward']==wrong['base_reward']
        c=correct_short_reward(1,1.,t,True)
        assert 0<c['correct_short_bonus']<=.5
        assert math.isclose(c['reward']-c['base_reward'],.5*256/(t+256),abs_tol=1e-12)
        zero=correct_short_reward(1,1.,t,True,correct_short_weight=0)
        assert zero['reward']==zero['base_reward']
    ts=[0,16,100,122,256,1000,1000000]
    cs=[correct_short_reward(1,1.,t,True) for t in ts]
    assert all(a['correct_short_bonus']>b['correct_short_bonus'] for a,b in zip(cs,cs[1:]))
    # Algebra: beta*EM*scale/(T+scale) = beta*EM - beta*EM*T/(T+scale).
    for em in [0,1]:
        for t in ts:
            v=correct_short_reward(em,float(em),t,True)
            alternative=(1+.5)*em+.25*em-(10+.5*em)*t/(t+256)+.02
            assert math.isclose(v['reward'],alternative,abs_tol=1e-12)
    groups={}
    for p in sorted((ROOT/'episodes').glob('*.json')):
        e=json.loads(p.read_text());v=correct_short_reward(e['em'],e['f1'],e['generated_tokens'],e['format_valid']);assert math.isclose(v['base_reward'],e['reward'],abs_tol=1e-12);groups.setdefault(e['qid'],[]).append(e|v)
    assert len(groups)==48 and sum(map(len,groups.values()))==192
    stats={};conflicts=[]
    for field in ['base_reward','reward']:
        n=0
        for q,es in groups.items():
            top=max(e[field] for e in es);best=[e for e in es if math.isclose(e[field],top,abs_tol=1e-12)]
            if any(e['em'] for e in es) and all(not e['em'] for e in best):n+=1
        stats[field+'_top_wrong_despite_correct_available']=n
    example=[]
    for em,t in [(0,16),(1,122)]:example.append(dict(em=em,t=t,**correct_short_reward(em,float(em),t,True)))
    report=dict(verified=True,scope='Formula/properties and192existingpath rescoring only; NO training or rollout for this bonus',length_weight=10,correct_short_weight=.5,stats=stats,example=example,properties=['wrong reward exactly unchanged','correct-only positive bonus','shorter correct gets larger bonus','beta0 exactly recovers old reward','base length term unchanged','algebra checked'],limitation='Adds both an EM bonus and extra correct-only length penalty; does not guarantee long-correct beats short-wrong at lambda10')
    out=REPO/'lab/reports/correct_short_reward_design_20261025';out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
