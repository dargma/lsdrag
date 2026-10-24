from lsdrag.decision_current_policy_targets import current_policy_targets

def row(q,anchor,values):
 samples=[dict(action=i,f1=f,generated=c,errors=[]) for i,(f,c) in enumerate(values)]
 return dict(qid=q,turn=0,legal=[True]*len(values)+[False]*(4-len(values)),reference_probs=[int(i==anchor) for i in range(4)],samples=samples)

def test_terminal_gain_can_spend_savings_without_sacrificing_other_quality():
 # A's useful extra step costs20; B has a quality-preserving30token saving.
 a=row('a',0,[(0,20),(1,40)])
 b=row('b',0,[(1,50),(1,20)])
 labels,e=current_policy_targets([a,b])
 assert labels=={0:1,1:1} and e['target_suffix_f1']>e['reference_suffix_f1']
 assert e['target_suffix_generated']<=e['reference_suffix_generated']

def test_no_quality_tradeoff_or_cheap_failure_or_errored_pair():
 a=row('a',0,[(1,100),(.5,1)])
 b=row('b',0,[(0,100),(0,1)])
 c=row('c',0,[(.5,100),(1,1)]);c['samples'][1]['errors']=['parse']
 labels,e=current_policy_targets([a,b,c]);assert labels=={}

def test_does_not_spend_unavailable_budget_or_modify_exact_tie():
 a=row('a',0,[(.5,10),(1,100)])
 b=row('b',0,[(1,10),(1,10)])
 labels,_=current_policy_targets([a,b]);assert labels=={}
