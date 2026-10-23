"""CPU-only importance prediction and exact-count acquisition controls."""
import hashlib
import math
import numpy as np

def family_components(qids,families):
    parent={q:q for q in qids}
    def root(q):
        while parent[q]!=q:
            parent[q]=parent[parent[q]];q=parent[q]
        return q
    owner={}
    for q in qids:
        for f in families[q]:
            if f in owner:parent[root(q)]=root(owner[f])
            else:owner[f]=q
    groups={}
    for q in qids:groups.setdefault(root(q),[]).append(q)
    return sorted([sorted(v) for v in groups.values()])

def balanced_folds(groups,counts,n=6,seed=20261027):
    if len(groups)<n:raise ValueError('Too few independent families')
    key=lambda g:(-sum(counts[q] for q in g),hashlib.sha256((str(seed)+':'+','.join(g)).encode()).hexdigest())
    folds=[[] for _ in range(n)];sizes=[0]*n
    for group in sorted(groups,key=key):
        j=min(range(n),key=lambda i:(sizes[i],i));folds[j]+=group;sizes[j]+=sum(counts[q] for q in group)
    return [sorted(f) for f in folds]

def fit_predict_ridge(train_x,train_y,train_qids,test_x,penalty=1.):
    """Standardization and all fitting use training rows only."""
    x=np.asarray(train_x,dtype=np.float64);y=np.asarray(train_y,dtype=np.float64)
    count={q:train_qids.count(q) for q in set(train_qids)}
    w=np.array([1/count[q] for q in train_qids]);w/=w.sum()
    mean=(w[:,None]*x).sum(0);scale=np.sqrt((w[:,None]*(x-mean)**2).sum(0)).clip(1e-6)
    z=(x-mean)/scale;ym=w@y;X=np.sqrt(w)[:,None]*z;Y=np.sqrt(w)*(y-ym)
    coef=X.T@np.linalg.solve(X@X.T+penalty*np.eye(len(x)),Y)
    pred=(np.asarray(test_x)-mean)/scale@coef+ym
    return np.maximum(pred,0),{'mean':mean,'scale':scale,'coef':coef,'intercept':ym}

def subset_counts(costs,budget):
    n=len(costs);f=[[0]*(budget+1) for _ in range(n+1)];f[0][0]=1
    for i,c in enumerate(costs):
        for b in range(budget+1):f[i+1][b]=f[i][b]+(f[i][b-c] if b>=c else 0)
    return f

def exact_budget(costs,fraction):
    b=math.floor(sum(costs)*fraction);f=subset_counts(costs,b)
    return max(i for i,v in enumerate(f[-1]) if v)

def uniform_inclusion(costs,budget):
    """Exact marginal inclusion over uniformly random feasible subsets."""
    f=subset_counts(costs,budget);reverse=subset_counts(list(reversed(costs)),budget);total=f[-1][budget]
    if not total:raise ValueError('Unreachable budget')
    p=[];n=len(costs)
    for i,c in enumerate(costs):
        num=sum(f[i][b]*reverse[n-i-1][budget-c-b] for b in range(budget-c+1))
        p.append(num/total)
    return np.array(p)

def select_exact(scores,costs,budget):
    dp=[None]*(budget+1);dp[0]=(0.,())
    for i,(score,cost) in enumerate(zip(scores,costs)):
        for b in range(budget,cost-1,-1):
            prev=dp[b-cost]
            if prev is None:continue
            val=prev[0]+float(score)
            if dp[b] is None or val>dp[b][0]+1e-12:dp[b]=(val,prev[1]+(i,))
    if dp[budget] is None:raise ValueError('Unreachable budget')
    mask=np.zeros(len(costs));mask[list(dp[budget][1])]=1
    assert int(mask@np.array(costs))==budget
    return mask
