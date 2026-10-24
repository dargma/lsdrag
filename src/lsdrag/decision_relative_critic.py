"""Action-relative quality learning with unchanged absolute generated-cost target."""
import torch

def relative_quality_loss(q,c,target_q,target_c,legal,weights):
    # Illegal/unobserved target NaNs are removed before pair subtraction.
    clean_q=torch.where(legal,target_q,torch.zeros_like(target_q))
    difference=q[:,:,None]-q[:,None,:]
    target=clean_q[:,:,None]-clean_q[:,None,:]
    pairs=(legal[:,:,None]&legal[:,None,:])&torch.ones_like(difference,dtype=torch.bool).triu(1)
    assert (pairs.sum((1,2))>0).all(), 'Relative critic requires two observed legal actions'
    qloss=torch.where(pairs,(difference-target).square(),0).sum((1,2))/pairs.sum((1,2))
    clean_c=torch.where(legal,target_c,c.detach())
    closs=((c-clean_c)/100).square().sum(-1)/legal.sum(-1)
    return ((qloss+closs)*weights).sum()
