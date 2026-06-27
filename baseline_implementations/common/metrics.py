#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance metrics for metric learning

Created on Mon Jun 12 17:16:51 2023

@author: jack
"""

import torch as T
import torch.nn.functional as F

# ---------------------------- Euclidean Distance -----------------------------


# Find Euclidean Distance Between Tensors
def euclidean_distance(a, b):
    """
    Finds Euclidean distance between tensors a and b
    """
    # calculate and return pairwise euclidean distance
    return T.sum(T.pow(a - b, 2), dim=-1).sqrt()
    # return F.pairwise_distance(a, b, dim = 0)


def euclidean_distance_matrix(embeddings, squared=True):
    """
    Calculate the distance matric containng distances between all embeddings in tensor

    Parameters
    ----------
    embeddings : Pytorch Tensor
        Tensor containing embeddings for distance calculation
    squared : Bool, optional
        Return squared euclidean distances if true. The default is True.

    Returns
    -------
    d : Pytorch Tensor
        distance matrix where element 2,1 is the distance between the first and second embeddings.

    """
    # find the dot product matrix for embeddings
    dot_product = T.matmul(embeddings, T.transpose(embeddings, 0, 1))

    # get digonals from dp matrix (squared sum of embedding row)
    sq_sum = T.diagonal(dot_product, 0)

    """
    calculate the distance matrix using: 
        ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    
    shape is shape (batch_size, batch_size)
    """

    d = T.unsqueeze(sq_sum, 0) - 2.0 * dot_product + T.unsqueeze(sq_sum, 1)
    d = T.clamp(d, min=0.0)  # prevent negatives due to floating point errors

    # sqrt distance matrix if squared distance not required
    if not squared:
        mask = T.eq(d, T.tensor(0.0)).float()  # find zero value s
        d = d + (mask * 1e-16)  # replace with small epsilon value

        d = T.sqrt(d)  # sqrt distance matrix

        d = d * (1.0 - mask)  # set epsilon values back to 0.0

    return d


# ----------------------------- Cosine Distance -------------------------------


# find cosine similarity between samples
def cosine_distance(a, b, chunk_size=None):
    """
    calculates cosine distance between tensors a and b
    """
    if chunk_size is None:
        # Calculate cosine distance and reutrn inverse
        return 1 - F.cosine_similarity(a, b, dim=-1)
    else:
        num_samples = a.size(0)
        class_dists = []
        for i in range(0, num_samples, chunk_size):
            # Get the current chunk of samples
            chunk = a[i : i + chunk_size]
            # Compute the cosine distance between the chunk and the centroid
            dists = cosine_distance(chunk, b.expand_as(chunk))
            # Append the distances to the list
            class_dists.append(dists)
        # Concatenate all the distance tensors
        return T.cat(class_dists, dim=0)


# calculate cosine distance matrix
def cosine_distance_matrix(embeddings, squared=False):
    embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cos_sim = T.matmul(embeddings_norm, embeddings_norm.t())
    distance_matrix = 1 - cos_sim
    if squared:
        distance_matrix = distance_matrix**2
    return distance_matrix


# ---------------------------- Manhattan Distance -----------------------------


# define mahanattan distance matrix
def manhattan_distance_matrix(embeddings, squared=False):
    distance_matrix = T.abs(embeddings.unsqueeze(1) - embeddings.unsqueeze(0))
    distance_matrix = T.sum(distance_matrix, -1)
    if squared:
        distance_matrix = T.pow(distance_matrix, 2)
    return distance_matrix


# ------------------------------- SNR Distance --------------------------------


def snr_distance(a, b, batched=True):
    """
    Pytorch impplementation of SNR distance metric (although not technically a metric since snr(a,b) != snr(b,a))
    netric from paper available at:

        https://arxiv.org/abs/1904.02616


    Parameters
    ----------
    a : Pytorch Tensor
        First input tensor for distance calculation.
    b : Pytorch Tensor
        Second input tensor for distance calculation.
    batched : Bool, optional
        Calculates the distance using matrix operations for model training if True. Otherewise
        calculates for two single data samples The default is True.

    Returns
    -------
    Pytorch Tensor
        Tensor containing pairwise SNR distance between input tensors.

    """
    # b is signal plus noise, a is signal

    n = b - a

    if batched:
        # remove bias from signal and noise to have mean of 0
        a = a - T.unsqueeze(T.mean(a, dim=1), 1)
        n = n - T.unsqueeze(T.mean(n, dim=1), 1)

        # find signal and noise powers
        a_pow = T.pow(a, 2)
        n_pow = T.pow(n, 2)

        var_a = T.sum(a_pow, dim=1)
        var_n = T.sum(n_pow, dim=1)

        return var_n / var_a  # return snr distance (1/snr)

    else:
        # remove bias from signal and noise to have mean of 0
        a = a - T.mean(a)
        n = n - T.mean(n)

        # find signal and noise powers
        a_pow = T.pow(a, 2)
        n_pow = T.pow(n, 2)

        var_a = T.sum(a_pow)
        var_n = T.sum(n_pow)

        return var_n / var_a  # return snr distance (1/snr)
