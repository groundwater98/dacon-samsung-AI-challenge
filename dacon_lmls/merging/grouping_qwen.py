import copy
import os
import pickle
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig

from dacon_lmls.utils.constants import FP32_EPS
from dacon_lmls.models.qwen.utils import QwenMoEWrapper
from dacon_lmls.merging.clustering import group_experts_by_clustering
from dacon_lmls.merging.overlap import compute_kl_divergence, get_prob_distributions, compute_wasserstein_distance

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1).item() / 2,
}

LEGAL_SIMILARITY_BASES = ["feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "random", "no",
                          "expert-output", "weight+expert-output", "router-logits+expert-output", "router-logits+weight+expert-output"]

# Utility to group experts in Qwen3 MoE layers and maintain per-layer state
# (group assignments, similarity matrices, usage frequencies, and initial centers)
class ExpertsGrouperForQwen3MoE(object):
    def __init__(
            self,
            config: Qwen3MoeConfig,
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
            start_layer: int = 0,
            group_limit: int = 4,
            data_limit: int = 1000000,
            random_start_center: bool = False,
            cluster: str = "hierarchical",
            linkage: str = "average",
            hierarchical_stopping_metric: str = "silhouette",
            overlap_metric: str = "cosine",
            dynamic_group: bool = False,
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.num_experts
        self.d_model = config.hidden_size
        self.d_ff = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.random_start_center = random_start_center
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

        self.sparse_layer_indices = list(range(start_layer, config.num_hidden_layers))
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self._init_center_state_dict = None
        self.reset_all()


    def reset_all(self):
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        self._init_center_state_dict = dict()
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts, device="cpu")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cpu") + torch.eye(self.num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    def load_init_center_state_dict(self, load_path: str):
        init_centers = pickle.load(open(load_path, "rb"))
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._init_center_state_dict[ffn_name] = torch.tensor(init_centers[layer_idx])

    def _assign_num_groups_per_layer(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        print("total_num_groups: ", total_num_groups)
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"

            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[ffn_name] = torch.ones_like(usage_frequency_dict[ffn_name])

            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            k = (self.num_experts // self.group_limit) + 1 if (self.num_experts % self.group_limit) != 0 else (self.num_experts // self.group_limit)

            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"Frequency threshold: {frequency_threshold}")

        if frequency_threshold == 1.0:
            raise ValueError("The number of groups is too large, please reduce the number of groups.")

        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups_per_layer[ffn_name] = torch.sum(
                (usage_frequency_dict[ffn_name] > frequency_threshold).long()
            ).item()

        return num_groups_per_layer
    
    #NOTE: Clustering
    def cluster_experts(
            self,
            model: Qwen3MoeForCausalLM,
            dataloader: DataLoader,
            num_groups: int,
    ):
        if self.similarity_base == "expert-output":
            # Perform clustering using expert activation as the similarity base
            dom_experts = self.group_experts_by_clustering_output(
                model=model,
                dataloader=dataloader,
                num_groups=num_groups
            )

        else:
            raise ValueError(f"Unknown similarity base: {self.similarity_base}")
        # Return representative experts
        return dom_experts
    
    # Calculates the similarity between experts, groups them, and returns the representative experts
    def group_experts_by_clustering_output(
        self,
        model: Qwen3MoeForCausalLM,
        dataloader: DataLoader,
        num_groups: int,
    ):
        model.eval()
        dom_experts = dict()
        forwarded_hidden_states = {}
        handles = []

        # Define forward hook to collect MoE layer input activations
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1])) # .cpu()
            return hook
        
        # Register hooks on each sparse MoE layer to capture activations
        for layer_idx in tqdm(self.sparse_layer_indices, desc=f"[Merging]Registering forward hook..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            moe = model.model.layers[layer_idx].mlp
            handles.append(moe.register_forward_hook(_get_activation_hook(ffn_name)))

        # Run the model over the dataloader to collect MoE inputs (activations)
        for batch in tqdm(dataloader, desc=f"Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        # Remove hooks and clear cache
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        # Dynamically assign the number of groups per layer based on expert usage frequency
        if self.dynamic_group:
            num_groups_per_layer = self._assign_num_groups_per_layer(
                num_groups, self.sparse_layer_indices
            )

        # For each sparse MoE layer: compute expert outputs and cluster them
        for layer_idx in tqdm(self.sparse_layer_indices, desc="Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            _device = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.device
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).to(_device) # .cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                expert_outputs = torch.stack(expert_outputs).to(torch.float32)

                num_groups_in_layer = num_groups_per_layer[ffn_name] if self.dynamic_group else num_groups
                dom_experts[ffn_name], label = group_experts_by_clustering(
                    model="qwen",
                    num_groups=num_groups_in_layer,
                    cluster=self.cluster,
                    linkage=self.linkage,
                    hierarchical_stopping_metric=self.hierarchical_stopping_metric,
                    num_experts=self.num_experts,
                    experts=expert_outputs,
                    init_center=self._init_center_state_dict[ffn_name] if ffn_name in self._init_center_state_dict else None)
                self._group_state_dict[ffn_name] = label.cpu()
            del layer_input
        torch.cuda.empty_cache()
        return dom_experts
    

    def compute_all_usages(
            self,
            model: Qwen3MoeForCausalLM,
            dataloader: DataLoader,
            mode: str = "frequency", # frequency, routing-score
    ):
        model.eval()
        config = model.config
        # Iterate through the dataloader to gather routing decisions
        for batch in tqdm(dataloader, desc=f"Evaluating routing distribution"):
            # Move batch tensors to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            all_router_logits = outputs.router_logits
            if mode == "frequency":
                all_router_logits = torch.stack(all_router_logits)  # of shape (num_hidden_layers, num_tokens, num_experts)
                selected_experts = torch.topk(all_router_logits, 2, dim=-1)[1].reshape(
                    config.num_hidden_layers, -1
                )  # of shape (num_hidden_layers, num_tokens * 2)
                for layer_idx in self.sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
                    self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
            else: # routing-score
                for layer_idx in self.sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    router_score = F.softmax(all_router_logits[layer_idx], dim=1)
                    scores = router_score.float().sum(0) / router_score.shape[0]
                    self._usage_frequency_state_dict[ffn_name] += scores.cpu()
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }
    
    def _compute_all_similarities_by_expert_outputs(
            self, model: Qwen3MoeForCausalLM, dataloader: DataLoader
    ):
        model.eval()
        forwarded_hidden_states = {} # Dict to store MoE layer inputs
        handles = []
        # Hook function to capture layer inputs
        def _get_activation_hook(name):
            def hook(module, input, output):
                # forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return hook
        
        # Register hooks on all sparse layers (MLPs)
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            # Save handle to later remove
            handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )

        # Run the model to collect inputs via hooks
        for batch in tqdm(dataloader, desc=f"Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        # torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    if self.overlap_metric == "cosine":
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                    else:
                        expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input))
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        if i == j:
                            self.save_similarity(ffn_name, i, j, 1.0)
                            continue
                        if self.overlap_metric == "kl-divergence":
                            p = get_prob_distributions(expert_outputs[i])
                            q = get_prob_distributions(expert_outputs[j])
                            similarity = compute_kl_divergence(p, q)
                        elif self.overlap_metric == "wasserstein": # wasserstein
                            similarity = compute_wasserstein_distance(expert_outputs[i], expert_outputs[j])
                        else: # cosine
                            i_flat = expert_outputs[i].flatten()
                            j_flat = expert_outputs[j].flatten()
                            similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
                        self.save_similarity(ffn_name, j, i, similarity)

@torch.no_grad()
def _merge_mlp_experts_by_usage_frequency_weighting(
        ffn,  # Qwen3MoeSparseMoeBlock
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
        pruning_ratio: float
):
    #new_experts = torch.nn.ModuleList()
    new_experts = []
    new_gate_weights = []
    group_freqs = []

    device = ffn.gate.weight.device
    dtype = ffn.gate.weight.dtype

    for label in group_labels.unique():
        # Find experts belonging to this cluster
        expert_indices = torch.where(group_labels == label)[0]

        denom = torch.sum(usage_frequencies[expert_indices]) + FP32_EPS

        # Initialize accumulators for weighted averaging
        gate_proj_weight = 0
        down_proj_weight = 0
        up_proj_weight = 0

        # Compute frequency-weighted average of expert parameters
        for idx in expert_indices:
            freq = usage_frequencies[idx]
            gate_proj_weight += ffn.experts[idx].gate_proj.weight * freq
            down_proj_weight += ffn.experts[idx].down_proj.weight * freq
            up_proj_weight   += ffn.experts[idx].up_proj.weight * freq

        gate_proj_weight /= denom
        down_proj_weight /= denom
        up_proj_weight   /= denom

        # Create a new expert with parameters replaced by frequency-weighted averages
        merged_expert = copy.deepcopy(ffn.experts[expert_indices[0]])
        merged_expert.gate_proj.weight.copy_(gate_proj_weight.to(dtype=dtype, device=device))
        merged_expert.down_proj.weight.copy_(down_proj_weight.to(dtype=dtype, device=device))
        merged_expert.up_proj.weight.copy_(up_proj_weight.to(dtype=dtype, device=device))
        
        # Add the merged expert to the new expert list
        new_experts.append(merged_expert)

        # Use the gate weight of the most frequently used expert
        rep_idx = expert_indices[torch.argmax(usage_frequencies[expert_indices])]
        rep_gate_weight = ffn.gate.weight[rep_idx].clone()
        new_gate_weights.append(rep_gate_weight)

        group_freqs.append(torch.max(usage_frequencies[expert_indices]).item())

    # Apply frequency-based pruning across merged experts
    new_experts, new_gate_weights = expert_prune_by_frequency(
        torch.nn.ModuleList(new_experts),
        new_gate_weights,
        group_freqs,
        pruning_ratio=pruning_ratio
    )

    # Rebuild the gate layer to match the reduced number of experts
    ffn.experts = new_experts

    in_features = ffn.gate.in_features
    new_gate = torch.nn.Linear(in_features, len(new_experts), bias=False).to(device=device, dtype=dtype)
    new_gate.weight.data.copy_(
        torch.stack(new_gate_weights, dim=0).to(device=device, dtype=dtype)
    )
    ffn.gate = new_gate

    return ffn

@torch.no_grad()
def merge_by_groups_with_usage_weighted(
        model: Qwen3MoeForCausalLM,
        grouper: ExpertsGrouperForQwen3MoE,
        pruning_ratio: float = 1.0,
        merging_layers: Optional[List[int]] = None
) -> Qwen3MoeForCausalLM:
    # Retrieve expert usage frequencies and group labels from the grouper
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()

    # Iterate over all sparse MoE layers
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"Merging experts with usage-frequency-weighted averaging..."
    ):
        # Skip layers not included in the merging_layers list
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        
        # Replace the MLP with a merged/pruned version using weighted averaging
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=usage_frequencies,
            pruning_ratio=pruning_ratio
        )
    return model    


def expert_prune_by_frequency(
    experts: torch.nn.ModuleList,
    gate_weights: list,
    group_freqs: list,
    pruning_ratio: float 
):

    group_freqs = torch.tensor(group_freqs)
    k = max(1, int(len(group_freqs) * pruning_ratio)) # Determine number of experts to keep (top-k)

    # Select indices of top-k experts with highest usage frequency
    keep_indices = torch.topk(group_freqs, k).indices.tolist()

    # Keep only the selected experts and their corresponding gate weights by keep_indices
    pruned_experts = [experts[i] for i in keep_indices]
    pruned_gate_weights = [gate_weights[i] for i in keep_indices]

    # Return pruned expert list and gate weights
    return torch.nn.ModuleList(pruned_experts), pruned_gate_weights
