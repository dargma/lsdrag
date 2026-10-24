from lsdrag.decision_pareto import cheaper_quality_pairs

def record(q0,q1,g0=(10,12),g1=(20,22)):
    return {'samples':[dict(action=a,rep=i,f1=q,generated=g,errors=[]) for a,qs,gs in [(0,q0,g0),(3,q1,g1)] for i,(q,g) in enumerate(zip(qs,gs))]}

def test_positive_quality_preserving_saving():
    assert cheaper_quality_pairs([record((.5,.5),(.5,.5))])==[(0,0,3)]

def test_no_zero_quality_collapse_or_uncertain_preference():
    assert not cheaper_quality_pairs([record((0,0),(0,0))])
    assert not cheaper_quality_pairs([record((.4,.8),(.5,.5))])
    assert not cheaper_quality_pairs([record((.5,.5),(.5,.5),(10,25))])

def test_errors_excluded():
    r=record((1,1),(.5,.5));r['samples'][0]['errors']=['failure']
    assert not cheaper_quality_pairs([r])
