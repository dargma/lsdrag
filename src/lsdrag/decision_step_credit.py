"""Prepared reward replay helpers; no optimizer, critic or inference changes."""
import numpy as np


def generated_by_step(episode):
    decisions=episode['decisions'];calls=episode['reader_calls']
    if len(calls)!=2*len(decisions):raise ValueError('Expected cached prefill/generation pairs')
    result=[]
    for i,d in enumerate(decisions):
        pre,gen=calls[2*i:2*i+2]
        if pre.get('pack_slots')!=37 or pre.get('generated_tokens',0)!=0:
            raise ValueError('Unsupported prefill accounting')
        n=gen['generated_tokens']
        if not isinstance(n,int) or isinstance(n,bool) or n<0:raise ValueError('Invalid generated count')
        if d.get('branch_intervention'):raise ValueError('Intervention data is not on-policy replay')
        result.append(n)
    if sum(result)!=episode['generated_tokens']:raise ValueError('Episode token total mismatch')
    return np.asarray(result,dtype=np.float64)


def token_returns(episode,coefficient=.0003,remaining=True):
    if not np.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid coefficient')
    q=float(episode['f1'])
    if not np.isfinite(q) or not 0<=q<=1:raise ValueError('Invalid quality')
    g=generated_by_step(episode)
    cost=np.cumsum(g[::-1])[::-1] if remaining else np.full(len(g),g.sum())
    return q-coefficient*cost


def group_step_advantages(episodes,scale,coefficient=.0003,remaining=True):
    """Identical LOO baseline and externally fixed scale for total/remaining arms.

    Keep forced-tail costs but omit forced actions. A train-wide frozen scale avoids
    exploding suffix advantages in groups with equal total rewards. This remains
    a practical surrogate with existing inverse-length trajectory weighting.
    """
    if len(episodes)<2 or not np.isfinite(scale) or scale<=0:raise ValueError('Invalid group/scale')
    if len({e['qid'] for e in episodes})!=1:raise ValueError('Mixed-question group')
    returns=[token_returns(e,coefficient,remaining) for e in episodes]
    totals=np.asarray([token_returns(e,coefficient,False)[0] for e in episodes])
    baseline=(totals.sum()-totals)/(len(episodes)-1)
    gated=max(float(e['f1']) for e in episodes)==0
    result=[]
    for e,r,b in zip(episodes,returns,baseline):
        keep=np.asarray([not d['forced'] for d in e['decisions']],dtype=bool)
        result.append(np.zeros(int(keep.sum())) if gated else ((r-b)/scale)[keep])
    return result
