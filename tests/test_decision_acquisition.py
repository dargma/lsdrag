import itertools
import numpy as np
from lsdrag.decision_acquisition import family_components,balanced_folds,fit_predict_ridge,exact_budget,uniform_inclusion,select_exact

def test_family_transitive_and_split_atomic():
    g=family_components(['a','b','c','d'],{'a':{1},'b':{1,2},'c':{2},'d':{3}})
    assert g==[['a','b','c'],['d']]
    f=balanced_folds(g,dict(a=2,b=2,c=2,d=2),n=2)
    assert any(set(x)=={'a','b','c'} for x in f)

def test_budget_and_random_exact_enumeration():
    costs=[6,8,6,8,6];scores=[.8,.4,.2,.9,.1];budget=exact_budget(costs,.5)
    feasible=np.array([v for v in itertools.product([0,1],repeat=5) if np.dot(v,costs)==budget])
    assert np.allclose(uniform_inclusion(costs,budget),feasible.mean(0))
    chosen=select_exact(scores,costs,budget)
    assert np.isclose(chosen@scores,max(feasible@np.array(scores)))

def test_heldout_features_do_not_change_fit_or_other_predictions():
    rng=np.random.default_rng(3);x=rng.normal(size=(8,4));y=rng.random(8);z=rng.normal(size=(2,4));q=[str(i//2) for i in range(8)]
    p,a=fit_predict_ridge(x,y,q,z);z[1]=1e9;p2,b=fit_predict_ridge(x,y,q,z)
    assert p[0]==p2[0]
    for k in a:assert np.array_equal(a[k],b[k])
