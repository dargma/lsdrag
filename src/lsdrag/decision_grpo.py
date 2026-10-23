"""Group-relative advantages for four tool-policy trajectories per question."""
import numpy as np

def group_advantages(rewards, epsilon=1e-6):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or len(r)<2 or not np.isfinite(r).all():raise ValueError('Invalid reward group')
    return (r-r.mean())/(r.std()+epsilon)
