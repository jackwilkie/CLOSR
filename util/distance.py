''' Functions for distance calculations
'''

import torch as T
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np

# -- cosine distance function
def cosdist(a, b = None):
    a_norm = F.normalize(a, p=2, dim=-1)
    
    if b is not None:
        b_norm = F.normalize(b, p=2, dim=-1)
    else:
        b_norm = a_norm.clone()
        
    # Compute cosine similarity
    similarity = T.mm(a_norm, b_norm.T)
    
    # Since cosine distance = 1 - cosine similarity
    cosine_distance = (1 - similarity)/2 # divide by 2 gives range [0, 1]
    
    return cosine_distance

# -- cosine distnace for 3d tensors
def batched_cosdist(x: Tensor) -> Tensor:
    # x: [B, C, D] → [C, B, D]
    x = x.permute(1, 0, 2)
    
    # Normalize along D
    x_norm = F.normalize(x, p=2, dim=-1)  # [C, B, D]

    # Compute cosine similarity using batch matmul:
    # [C, B, D] @ [C, D, B] → [C, B, B]
    sim = T.bmm(x_norm, x_norm.transpose(1, 2))

    # Cosine distance = (1 - sim)/2
    dist = (1 - sim) / 2

    return dist  # [C, B, B]

# -- cosine distance computed in chunks
def chunked_centroid_sims(
    embeddings,
    centroid,
    chunk_size: Optional[int] = 1024,
):        
    if chunk_size is None:
        centroid_sims = (F.cosine_similarity(centroid.unsqueeze(0), embeddings, dim = 1)).squeeze().cpu().detach().numpy()
    else:
        # Assume centroid and embeddings are already defined
        num_chunks = (embeddings.size(0) + chunk_size - 1) // chunk_size  # Calculate the number of chunks
        centroid_sims = []

        for i in range(num_chunks):
            # Get the current chunk of embeddings
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, embeddings.size(0))
            val_chunk = embeddings[start_idx:end_idx]
            
            # Calculate cosine similarity for the current chunk
            chunk_sims = (F.cosine_similarity(centroid.unsqueeze(0), val_chunk, dim=1)).squeeze()
                
            # Detach and move to CPU, then convert to numpy
            centroid_sims.append(chunk_sims.cpu().detach().numpy())

        # Concatenate all chunks to get the full result
        centroid_sims = np.concatenate(centroid_sims)

    return centroid_sims

# -- closr distance measurements
def get_class_centroid(train_embeddings: Tensor, train_labels: Tensor, target_class: int) -> Tensor:
    """ Calculate C x D tensor, where C is the number of classes and d is the output dimensionality of the model
        Each C is the centroid of 

    Args:
        train_embeddings (Tensor): Embedded representations of the training data
        train_labels (Tensor): Training Labels
        target_class (int): _description_

    Returns:
        Tensor: _description_
    """
    ref_points = train_embeddings[:,target_class,:]
    ref_points = ref_points[train_labels == target_class]
    return T.mean(F.normalize(ref_points,dim=-1), dim = 0)

def get_centroids(train_embeddings: Tensor, train_labels: Tensor) -> Tensor:
    """Calculate C x D tensor, where C is the number of classes and d is the output dimensionality of the model
        Each C is the centroid of of class calculated from respective clad head.

    Args:
        train_embeddings (Tensor): B x C x D, closr embeddings
        train_labels (Tensor): Class labels for train embeddings

    Returns:
        Tensor: C x D, Class centroids
    """
    
    # infer number of classes from train_embeddings dimension
    n_classes = train_embeddings.size(1)
    
    # calculate centroid for each class
    centroids = T.stack([
        get_class_centroid(
            train_embeddings = train_embeddings, 
            train_labels = train_labels,
            target_class = c,
        )
    for c in range(n_classes)])
    
    return centroids

def calc_class_sims_chunked(centroids: Tensor, embeddings: Tensor, chunk_size: int = 1024) -> Tensor:
    B = embeddings.size(0)
    sims_chunks = []

    # Normalize centroids once
    centroids = F.normalize(centroids, dim=-1)
    centroids = centroids.unsqueeze(0)  # [1, C, D]

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = embeddings[start:end]
        chunk = F.normalize(chunk, dim=-1)
        sims = T.sum(chunk * centroids, dim=2)
        sims_chunks.append(sims.cpu().detach())
    return T.cat(sims_chunks, dim=0)  # [B, C]
