import os
import gc
import sys
import time
import pickle
from typing import Optional

import logging
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from dacon_lmls.evaluation import evaluate_fewshot, get_calib_dataloder
from dacon_lmls.merging.grouping_qwen import ExpertsGrouperForQwen3MoE, merge_by_groups_with_usage_weighted

logger = logging.getLogger(__name__)

class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen3-30B-A3B",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "freq",
        mode: Optional[str] = "normal",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        try_oracle: Optional[bool] = False,
        random_start_center: Optional[bool] = False,
        weight: Optional[str] = None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.try_oracle = try_oracle
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=4,
    )

def get_grouper(args, config):
    return ExpertsGrouperForQwen3MoE(
                config=config,
                similarity_base=args.similarity_base,
                start_layer=args.start_layer,
                group_limit=args.group_limit,
                data_limit=args.data_limit,
                random_start_center=args.random_start_center,
                cluster=args.cluster,
                linkage=args.linkage,
                dynamic_group=args.dynamic_group,
            )

def evaluation(args, model, tokenizer):
    result_dir = args.result_path.split("/")[:-1]
    result_dir = "/".join(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # if eval_ppl:
    #     evaluate_minipile_perplexity(
    #         model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
    #     )

    if isinstance(args.task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=args.task, num_fewshot=args.num_fewshot, output_path=args.result_path, log=True
        )
    else:
        for i, t in enumerate(args.task):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=args.num_fewshot, eval_batch_size=args.eval_batch_size, output_path=args.result_path, log=True
            )

def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()


def main(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen3-30B-A3B",
        dominant: Optional[str] = "no", # random, frequency, knowledge
        similarity_base: Optional[str] = "expert-output",
        merge: Optional[str] = "freq", 
        mode: Optional[str] = "normal", 
        n_sentences: Optional[int] = 128,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 16,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 1000000,
        random_start_center: Optional[bool] = False,
        num_fewshot: Optional[int] = 0,
        cluster: Optional[str] = "hierarchical",
        linkage: Optional[str] = "average",
        dynamic_group: Optional[bool] = False,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {similarity_base} grouping + {merge} merge - {mode}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}")
    
    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        dynamic_group=dynamic_group,
    )
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    # local_model_path = "/home/jisoo0204/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"

    # tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = AutoModelForCausalLM.from_pretrained(
    #     local_model_path,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )

    if model_path:
        model.load_state_dict(torch.load(model_name))
    model.eval()
    dataloader_for_merging = get_dataloader(args, tokenizer)
    grouper = get_grouper(args, model.config)

    print("Number of parameters before merging:", model.num_parameters())
    print(f"Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge == "freq" or dominant == "frequency":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)

    ### 2. Get dominant experts
    dom_experts = None
    dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
      

    ### 3. Merging
    if merge == "freq":
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )

    print(f"Merging time: {time.time() - group_st:.2f} seconds")

    ### 4. Grouping results
    print(f"======== Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        if dom_experts is None:
            print(f"Group {name}: {state.tolist()}")
        else:
            print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]}, {len(dom_experts[name])})")

    del grouper
    
    ### 5. Save model
    print("Number of parameters after merging:", model.num_parameters())
    model.config.num_experts = args.num_average_groups
    model.config.architectures = ["Qwen3MoeForCausalLM"]
    model.config.model_type = "qwen3_moe"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # torch.save(model.state_dict(), output_path+"/model.pth")

    model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)

    ### 6. Evaluation
    evaluation(args, model, tokenizer)


if __name__ == "__main__":
    Fire(main)
