''' Contrastive Learning for Open Set Recognition Loss
'''

import torch as T
from torch import Tensor
from .clad_loss import CLADLoss

import torch as T
from torch import Tensor
import torch.nn.functional as F

# -- loss function
def loss_calc(
    dists: Tensor, # B x B distance matrix
    y: Tensor, # class labels
    target_class: int,  # target class
    margin,
    eps = 1e-6,
    squared = True,
    alpha = 0.5,
    return_frac_pos = False
) -> Tensor:
    """ functional classwise DDOS loss calculation
    """
    B = dists.size(0)

    #get pair masks
    eq_pairs =  T.eq(T.unsqueeze(y, 0), T.unsqueeze(y, 1)) # get similar pair mask
    sim_mask = T.logical_and(eq_pairs, T.unsqueeze(y, 0) == target_class)  # mask for similar benign pairs
    dissim_mask = T.logical_and(~eq_pairs, T.unsqueeze(y, 1) == target_class)  # mask for dissimilar pairs where first sample is zero

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
    
    #loss = loss / B
    if return_frac_pos:
        return loss, fraction_positive_pairs
    else:
        return loss


class CLOSRLoss(CLADLoss):
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
        
        B, C, D = x.size() # input is batch x n_classes x d
        
        # normalise vevctors
        x =  F.normalize(x, p=2, dim=-1)
        
        # calculate distance matrix
        x_t = x.transpose(0, 1)  # C x B x D
        distance_matrix = T.bmm(x_t, x_t.transpose(1, 2))  # C x B x B
        distance_matrix = (1 - distance_matrix)/2
        
        self.fraction_positive_pairs = T.tensor(0.0)
        loss = T.mean(
            T.stack([
                loss_calc(dists = distance_matrix[c], y = y, target_class=c, margin = self.m)    
            for c in range(C)])
        )
        return loss