''' Contrastive learning for anomlay detection loss function
'''

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
from util.distance import cossim

def clad_calc(
    x1: Tensor,
    target_class: int,
    y: Tensor,
    x2: Optional[Tensor] = None,
    margin: Optional[float] = 1.0,
    distance_metric: Callable[[Tensor, Optional[Tensor]], Tensor] = cossim,
    return_frac_pos: bool = False,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> Tensor:
    """ Functional CLAD Loss
    """
    B = x1.size(0)
    
    #get pair masks
    eq_pairs =  T.eq(T.unsqueeze(y, 0), T.unsqueeze(y, 1)) # get similar pair mask
    sim_mask = T.logical_and(eq_pairs, T.unsqueeze(y, 0) == target_class)  # mask for similar benign pairs
    dissim_mask = T.logical_and(~eq_pairs, T.unsqueeze(y, 1) == target_class)  # mask for dissimilar pairs where first sample is zero
    
    dists = distance_metric(x1, x2) # get cossine distance matrix
    
    # get dists for similar pairs
    sim_dists = dists * sim_mask 
    
    # get dists fo dissimilar pairs
    if margin is not None:
        margin_tensor = T.full(dists.shape, margin, device = dists.device, dtype = dists.dtype)
        margin = margin
    else:
        margin_tensor = T.zeros(dists.shape, device = dists.device, dtype = dists.dtype)
        margin = 0
                
    dissim_dists = T.where(dissim_mask, dists, margin_tensor)

    # calculate loss
    n_sim = T.sum(T.greater(sim_dists, eps).float()) # number of similar pairs
    
    if squared:
        sim_dists = T.pow(sim_dists,2)
    
    sim_loss = T.sum(sim_dists)/(n_sim + eps) # mean loss for similar pairs

    dissim_loss = F.relu(margin - dissim_dists)
    
    if squared:
        dissim_loss = T.pow(dissim_loss,2)
        
    n_dissim = T.sum(T.greater(dissim_loss, eps).float())
    dissim_loss = T.sum(dissim_loss)/(n_dissim + eps)

    fraction_positive_pairs= n_dissim/ (T.sum(dissim_mask) + eps)
    loss = ((alpha) * sim_loss) + ((1-alpha) * dissim_loss)
    
    if return_frac_pos:
        return loss, fraction_positive_pairs
    else:
        return loss


class CLADLoss(nn.Module):
    def __init__(
        self, 
        m = 1.,
        squared = True,
        alpha = 0.5,
        eps = 1e-8,
        **kwargs,
    ):
        super().__init__()
        self.m = m 
        self.squared = squared
        self.alpha = alpha 
        self.fraction_positive_pairs = 0.
        self.eps = eps
        self.distance_metric = cossim
        
    def forward(
        self, 
        x: Tensor,
        y: Tensor,
        x2: Optional[Tensor] = None,
    ) -> Tensor:
        
        loss, frac_pos = clad_calc(
            x1 = x,
            target_class = 0,
            x2 = x2,
            y = y,
            margin = self.m,
            distance_metric = self.distance_metric,
            return_frac_pos = True,
            squared = self.squared,
            alpha = self.alpha,
            eps = self.eps,
        )
        
        self.fraction_positive_pairs = frac_pos
        return loss

    def get_fraction_pos(self):
        return self.fraction_positive_pairs