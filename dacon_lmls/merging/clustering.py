# -*- coding: utf-8 -*-

import torch
from typing import Optional

@torch.no_grad()
def group_experts_by_clustering(
    model: str,
    num_groups: int,
    cluster: str,
    linkage: str,
    hierarchical_stopping_metric: str,
    num_experts: int,
    experts: torch.Tensor,
    experts2: Optional[torch.Tensor] = None,
    experts3: Optional[torch.Tensor] = None,
    init_center: Optional[torch.Tensor] = None,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
):
    # Ensure tensors are float for distance computations
    experts = experts.to(torch.float)
    experts2 = experts2.to(torch.float) if experts2 is not None else None
    experts3 = experts3.to(torch.float) if experts3 is not None else None
    if cluster == "hierarchical":
        labels, dom_experts = hierarchical_clustering(experts, num_groups, linkage)
        print(f"group: {labels}, dom: {dom_experts}")
        return dom_experts, labels
        
    # Z-score standardization per feature, then shift to make min >= 0
    def _standardize(x):
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        min_value = x.min()
        return x - min_value

    # Initialize cluster centers from provided indices
    if init_center is not None:
        indices = init_center

    centers = experts[indices]
    centers2 = experts2[indices] if experts2 is not None else None
    centers3 = experts3[indices] if experts3 is not None else None
    distances = None
    assignments = None
    print(f"initial center: {centers.shape} {indices}")

    # Feature-dimension scales for normalization across views
    s1 = experts.shape[1]
    s2 = experts2.shape[1] if experts2 is not None else 1.0
    s3 = experts3.shape[1] if experts3 is not None else 1.0

    for _ in range(100):
        distances1 = _standardize(torch.cdist(experts, centers) / s1)
        distances2 = _standardize(torch.cdist(experts2, centers2) / s2) if experts2 is not None else torch.zeros(1, device=experts.device)
        distances3 = _standardize(torch.cdist(experts3, centers3) / s3) if experts3 is not None else torch.zeros(1, device=experts.device)
        print(f"distances1: {distances1.shape} {distances1}")
        print(f"distances2: {distances2.shape} {distances2}")
        print(f"distances3: {distances3.shape} {distances3}")

        distances = (w1 * distances1 + w2 * distances2 + w3 * distances3) / (w1 + w2 + w3)
        assignments = torch.argmin(distances, dim=1)
        del distances, distances1, distances2, distances3

        new_centers = torch.stack([experts[assignments == k].mean(dim=0) for k in range(num_groups)])
        new_centers2 = torch.stack([experts2[assignments == k].mean(dim=0) for k in range(num_groups)]) if experts2 is not None else None
        new_centers3 = torch.stack([experts3[assignments == k].mean(dim=0) for k in range(num_groups)]) if experts3 is not None else None
        
        print(f"assignments: {assignments}")
        for k in range(num_groups):
            print(f"cluster {k} {experts[assignments==k]}")
        print(f"new_centers: {torch.sum(torch.isnan(new_centers))}, {new_centers[0]}")
        if experts2 is not None:
            print(f"new_centers2: {torch.sum(torch.isnan(new_centers2))}, {new_centers2[0]}")
        
        # Convergence check: track maximum absolute shift among centers
        max_diff = 0
        for i in range(num_groups):
            diff = torch.max(torch.abs(new_centers[i] - centers[i]))
            diff2 = torch.max(torch.abs(new_centers2[i] - centers2[i])) if experts2 is not None else torch.zeros(1, device=experts.device)
            diff3 = torch.max(torch.abs(new_centers3[i] - centers3[i])) if experts3 is not None else torch.zeros(1, device=experts.device)
            max_diff = max(max_diff, diff.item(), diff2.item(), diff3.item())
            if max_diff > 0.1:
                print(f"diff: {diff.item()}, {diff2.item()}, {diff3.item()}")
        if max_diff < 1e-4:
            print("Converged!")
            break
        centers = new_centers
        centers2 = new_centers2 if experts2 is not None else None
        centers3 = new_centers3 if experts3 is not None else None
    
    # After convergence: pick a representative (closest) member per group
    center_indices = []
    for k in range(num_groups):
        cluster_members = experts[assignments == k]
        cluster_members2 = experts2[assignments == k] if experts2 is not None else None
        cluster_members3 = experts3[assignments == k] if experts3 is not None else None
        distances1 = torch.cdist(cluster_members, new_centers[k].unsqueeze(0))
        distances2 = torch.cdist(cluster_members2, new_centers2[k].unsqueeze(0)) if experts2 is not None else torch.zeros(1, device=experts.device)
        distances3 = torch.cdist(cluster_members3, new_centers3[k].unsqueeze(0)) if experts3 is not None else torch.zeros(1, device=experts.device)
        final_distances = (distances1 + distances2 + distances3) / 2

        closest_expert_idx = torch.argmin(final_distances, dim=0)
        center_indices.append(torch.where(assignments == k)[0][closest_expert_idx].item())
    # centers = experts[center_indices]
    del centers, centers2, centers3
    print(f"group: {assignments.cpu()}, dom: {center_indices}")
    return center_indices, assignments

# Compute pairwise distances
@torch.no_grad()
def compute_distance(pair_distances, clusters, method='average', X=None):
    # Average-linkage: distance between clusters is the mean of all pairwise distances
    if method == 'average':
        # dist(cluster i, cluster j) = sum_{x in cluster i, y in cluster j} dist(x, y) / (|cluster i| * |cluster j|)
        cluster_labels = torch.unique(clusters)
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        # Iterate through all pairs of clusters (ci, cj)
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                dist = []
                # Iterate through all pairs of points (vi, vj) for vi in ci and vj in cj
                for vi in torch.where(clusters == ci)[0]:
                    for vj in torch.where(clusters == cj)[0]:
                        dist.append(pair_distances[vi, vj].item())
                new_dist = torch.sum(torch.tensor(dist)) / (torch.sum(clusters == ci) * torch.sum(clusters == cj))
                distances[i, j] = new_dist
                distances[j, i] = new_dist
        distances.fill_diagonal_(float('inf'))
        # Find the minimum entry and map flat index back to (i, j)
        idx = torch.argmin(distances)
        final_i, final_j = cluster_labels[idx // distances.shape[0]], cluster_labels[idx % distances.shape[0]]
    else:
        raise NotImplementedError("Unsupported linkage method: {}".format(method))
    
    return final_i, final_j

@torch.no_grad()
def pairwise_distances(X, method='average'):
    """Compute pairwise Euclidean distances between points."""
    # Use (xi - xj)^2 = ||xi||^2 + ||xj||^2 - 2 xi^T xj
    dot_product = torch.mm(X, X.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0).sqrt()
    # For average-linkage, disallow self-matches by marking diagonal as +inf
    if method == 'average':
        distances.fill_diagonal_(float('inf'))
    return distances

@torch.no_grad()
def linkage_step(distances, pair_distances, clusters=None, method='average', X=None):
    
    ### 1. Find the pair of clusters with the smallest distance
    if method == 'average':
        i, j = compute_distance(pair_distances, clusters, method, X)
    
    if i > j:
        i, j = j, i
    
    if method == 'average':
        return i, j, distances
    
    return i, j, distances

@torch.no_grad()
def hierarchical_clustering(X, n_clusters, method='average'):
    """Perform hierarchical clustering using the specified linkage method."""
    print("hierarchical clustering - {} to {} clusters".format(method, n_clusters))
    device = X.device
    n_samples = X.shape[0]
    
    # Compute pairwise distances
    distances = pairwise_distances(X, method)
    pair_distances = distances.clone()
    
    # Initialize clusters
    clusters = torch.tensor([i for i in range(n_samples)])
    
    # Perform clustering
    while len(torch.unique(clusters)) > n_clusters:
        # Find the closest pair of clusters to merge
        i, j, distances = linkage_step(distances, pair_distances, clusters, method, X)
        print(f"clusters: {len(torch.unique(clusters))}, merge ({i}, {j})")
        cj = clusters[j]
        # Merge cluster j to cluster i
        clusters[clusters == cj] = clusters[i]

    # Reassign cluster IDs to be contiguous
    d = {}
    element_id = 0
    for i, idx in enumerate(clusters):
        if idx.item() not in d:
            d[idx.item()] = element_id
            element_id += 1
        clusters[i] = d[idx.item()]
    
    # Select representative center sample for each cluster
    center_indices = []
    for k in range(n_clusters):
        cluster_members = X[clusters == k]
        cluster_center = cluster_members.mean(dim=0)
        distances = torch.cdist(cluster_members, cluster_center.unsqueeze(0), p=2)
        closest_expert_idx = torch.argmin(distances, dim=0).item()
        center_indices.append(torch.where(clusters == k)[0][closest_expert_idx].item())
    
    del distances
    return clusters, center_indices

