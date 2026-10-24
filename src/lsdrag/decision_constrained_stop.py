"""Convex stopping fit with explicit retained-action and categorical-KL constraints."""
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

def fit_constrained_stop(features, margins, targets, loss_weights, protected,
                         final_logodds, state_weights, kl_limit=.01):
    x=np.asarray(features,dtype=np.float64)
    m=np.asarray(margins,dtype=np.float64)
    t=np.asarray(targets,dtype=np.float64)
    w=np.asarray(loss_weights,dtype=np.float64)
    odds=np.asarray(final_logodds,dtype=np.float64)
    sw=np.asarray(state_weights,dtype=np.float64)
    idx=np.asarray(protected,dtype=int)
    assert x.shape[0]==len(m)==len(t)==len(w)==len(odds)==len(sw)
    assert np.isclose(w.sum(),1) and np.isclose(sw.sum(),1)
    sign=np.where(m[idx]>0,1.,-1.)
    matrix=sign[:,None]*x[idx]
    floor=np.minimum(np.abs(m[idx]),.01)-np.abs(m[idx])
    def loss(d):
        z=m+x@d
        value=np.sum(w*(np.logaddexp(0,z)-t*z))+.0005*np.dot(d,d)
        grad=x.T@(w*(expit(z)-t))+.001*d
        return value,grad
    def kl(d):
        shift=x@d
        value=np.sum(sw*(np.logaddexp(0,odds+shift)-np.logaddexp(0,odds)-expit(odds)*shift))
        grad=x.T@(sw*(expit(odds+shift)-expit(odds)))
        return value,grad
    result=minimize(lambda d:loss(d)[0],np.zeros(x.shape[1]),jac=lambda d:loss(d)[1],method='SLSQP',
        constraints=[LinearConstraint(matrix,floor,np.inf),dict(type='ineq',fun=lambda d:kl_limit-kl(d)[0],jac=lambda d:-kl(d)[1])],
        options=dict(maxiter=500,ftol=1e-10))
    violation=max(0.,float(np.max(floor-matrix@result.x)))
    evidence=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        kl=float(kl(result.x)[0]),constraint_violation=violation,loss_before=float(loss(np.zeros(x.shape[1]))[0]),loss_after=float(result.fun))
    return result.x,evidence
