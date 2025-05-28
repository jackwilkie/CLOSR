''' Contrastive Learning for Open Set Recognition Loss
'''

import torch as T
from torch import Tensor
from typing import Union, Tuple, Optional, Callable

from .clad_loss import contrastive_anomaly, clad_calc
from .utils import TripletLossWrapper, batched_cosdist

def closr_loss(
    x: Tensor,
    y: Tensor,
    margin: Optional[float] = 1.0,
    distance_metric: Callable[[Tensor, Optional[Tensor]], Tensor] = batched_cosdist,
    return_frac_pos: bool = False,
    squared: bool = True,
    eps: float = 1e-8,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    if x.ndim != 3:
        raise ValueError(f'Three dimensional input tensor expected, got: {x.size()}')
        
    # compute distance matrix
    dists = distance_metric(x) # C x B x B distance matrix
    
    if squared:
        dists = T.pow(dists, 2)
    
    loss_tuples = [
        clad_calc(
            dists = dists[c],
            target_class = c,
            y = y,
            margin = margin,
            return_frac_pos = True,
            eps = eps,
    )
    for c in range(x.size(1))] # calculate loss for each head
    
    # Unzip the tuples into two separate lists
    loss_elements, frac_pos_elements = zip(*loss_tuples)
    loss = T.sum(T.stack(loss_elements))/x.size(1)
    frac_pos = T.mean(T.stack(frac_pos_elements))
    
    if return_frac_pos:
        return loss, frac_pos
    else:
        return loss

    
class CLOSRLoss(contrastive_anomaly.ContrastiveAnomaly):
    def __init__(
        self, 
        *args,
        n_classes: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        
    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        
        if x.ndim != 3 and x.size(1) != self.n_classes:
            raise ValueError(f'Input invalid size. Got: {x.size()}, expected: B x {self.n_classes} x d')
        
        loss, frac_pos = closr_loss(
            x = x,
            y =y,
            margin = self.m,
            return_frac_pos = True,
            squared = self.squared,
            eps = self.eps,
        )
        
        self.fraction_positive_pairs = frac_pos
        return loss

def closr_loss(*args, **kwargs) -> TripletLossWrapper:
    return TripletLossWrapper(CLOSRLoss, *args, **kwargs)
