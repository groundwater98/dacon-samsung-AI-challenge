# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

def get_prob_distributions(expert_outputs):
    return F.softmax(expert_outputs, dim=-1)  # Normalize across the last dimension


def compute_kl_divergence(p, q):
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return - (F.kl_div(p.log(), q, reduction='batchmean') + 
            F.kl_div(q.log(), p, reduction='batchmean')) / 2


def compute_wasserstein_distance(p, q):
    # Ensure the tensors are on the same device and sort them
    p_sorted, _ = torch.sort(p)
    q_sorted, _ = torch.sort(q)
    
    # Compute the Wasserstein distance as the mean absolute difference
    # between the sorted values (CDF difference in a way)
    wasserstein_dist = torch.mean(torch.abs(p_sorted - q_sorted))
    
    return -wasserstein_dist
