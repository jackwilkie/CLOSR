''' Contrastive learning for anomlay detection loss function
'''

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Union, Tuple
from util.distance import cosdist

# -- loss function
def clad_calc(
    dists: Tensor,
    target_class: int,
    y: Tensor,
    margin: Optional[float] = 1.0,
    return_frac_pos: bool = False,
    eps: float = 1e-8,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    ''' Functional CLAD calculation
    '''
    #get pair masks
    eq_pairs =  T.eq(T.unsqueeze(y, 0), T.unsqueeze(y, 1)) # get similar pair mask
    dissim_mask = T.logical_and(~eq_pairs, T.unsqueeze(y, 1) == target_class)  # mask for dissimilar pairs where first sample is zero
    
    # get dists fo dissimilar pairs
    if margin is not None:
        margin_tensor = T.full(dists.shape, margin, device = dists.device, dtype = dists.dtype)
        margin = margin
    else:
        margin_tensor = T.zeros(dists.shape, device = dists.device, dtype = dists.dtype)
        margin = 0
                
    dissim_dists = T.where(dissim_mask, dists, margin_tensor)
    dissim_loss = F.relu(margin - dissim_dists)
    
    n_dissim = T.sum(T.greater(dissim_loss, eps).float())
    loss = T.sum(dissim_loss)/(n_dissim + eps)

    fraction_positive_pairs= n_dissim/ (T.sum(dissim_mask) + eps)
    
    if return_frac_pos:
        return loss, fraction_positive_pairs
    else:
        return loss
    
    
def clad_loss(
    x: Tensor,
    target_class: int,
    y: Tensor,
    margin: Optional[float] = 1.0,
    distance_metric: Callable[[Tensor, Optional[Tensor]], Tensor] = cosdist,
    return_frac_pos: bool = False,
    squared: bool = True,
    eps: float = 1e-8,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """ Functional CLAD Loss
    """
    
    dists = distance_metric(x, x) # get cossine distance matrix
    if squared:
        dists = T.pow(dists, 2)
    
    return clad_calc(
        dists = dists,
        target_class = target_class, 
        y = y,
        margin = margin,
        return_frac_pos = return_frac_pos,
        eps = eps,
    )
    

class CLADLoss(nn.Module):
    def __init__(
        self, 
        m: float = 1.,
        squared: bool = True,
        distance_metric: Callable[[Tensor, Tensor], Tensor] = cosdist,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.m = m 
        self.squared = squared
        self.fraction_positive_pairs = 0.
        self.eps = eps
        self.distance_metric = distance_metric
        
    def forward(
        self, 
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        
        loss, frac_pos = clad_loss(
            x = x,
            target_class = 0,
            y = y,
            margin = self.m,
            distance_metric = self.distance_metric,
            return_frac_pos = True,
            squared = self.squared,
            eps = self.eps,
        )
        
        self.fraction_positive_pairs = frac_pos
        return loss
    
    def get_fraction_pos(self):
        return self.fraction_positive_pairs