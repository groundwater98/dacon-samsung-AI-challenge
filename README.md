# 2025 Samsung AI Challenge Framework

Our method reduces the number of experts in large-scale Sparsely activated Mixture-of-Experts (SMoE) models without retraining.  
The key idea is to **merge functionally similar experts** by analyzing their output behavior, rather than depending on routing decisions.

## ⚙️ Software Requirements 
> This code was developed and tested with  <img src="https://pytorch.org/assets/images/logo-icon.svg" alt="PyTorch" width="20" height="20">  PyTorch 2.7.1.

* **Step 1: Make sure you have Python and pip (or conda) installed on your system.**
  ```bash
  conda create -n lmls python=3.10 -y
  conda activate lmls
  ```
* **Step 2: Install dependencies.**
   ```bash
   pip install -r requirements.txt
   pip install lm-eval
   ```
  - https://github.com/EleutherAI/lm-evaluation-harness.git

* **Step 3: Install C4 datasets (calibration datasets).**
   ```bash
   wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz -P dacon_lmls/data/
   ```
   ```bash
   gunzip -k dacon_lmls/data/c4-train.00000-of-01024.json.gz
   ```

## 📝 Instructions
### 1. Merging Process

1. **Expert Output Collection**  
   - For each expert in the MoE layer, we collect its output representations using a shared set of input samples.  
   - These outputs act as a functional “signature” that reflects the behavior of the expert.

2. **Similarity Computation**  
   - Compute pairwise similarities (Euclidean distance) between experts based on their output vectors.  
   - This step identifies experts that behave alike, regardless of routing frequency.

3. **Hierarchical Clustering**  
   - Apply hierarchical agglomerative clustering on the similarity scores to group experts into clusters.  
   - Functionally closer experts are merged earlier in the clustering tree.

4. **Expert Merging**  
   - Within each cluster, merge expert weights (e.g., projection matrices) using weighted averaging.  
   - This reduces the total number of experts while maintaining functional diversity.
---

### 2. Running Experiments

We provide ready-to-use scripts for running experiments.  
In particular, the file `./scripts/qwen/run.sh` contains the command-line setup for our framework.

You can either:

1. **Modify the script** to adjust parameters (e.g., model path, number of experts, clustering options) for your own experiments.
2. **Run it directly** to reproduce our default setup.
   ```bash
   bash scripts/qwen/run.sh
   ```

---
### 3. Output

- **Model & Tokenizer**  
  - The merged model and tokenizer are saved in  
    `./results/qwen/merge/test`.

- **Result Log**  
  - The experimental results log is saved in  
    `./results/log_test`.

## 🔎 Command-line Arguments
- **task**  
  The evaluation tasks to run. Multiple tasks can be specified as a comma-separated list.

- **model_name**  
  The base model to evaluate on. For example, `"Qwen/Qwen3-30B-A3B"`.

- **dominant**  
  Strategy for selecting dominant experts in each group.  
  - `"no"`: Do not pre-select dominant experts; use clustering instead.

- **similarity_base**  
  The metric used to measure similarity between experts when grouping. `expert-output`.

- **cluster**  
  The clustering algorithm applied when grouping experts. `hierarchical`.

- **linkage**  
  Linkage method used in hierarchical clustering. Options: `average`.

- **merge**  
  The merging method for grouped experts.
  - `"freq"`: Frequency-weighted merging.

- **num_average_groups**  
  The number of experts to keep per layer after merging.

- **n_sentences**  
  Number of sentences sampled from the dataset (e.g., C4) for statistics collection.

- **train_batch_size**  
  Batch size used during statistics collection for grouping/merging (no gradient updates).

- **eval_batch_size**  
  Batch size used during evaluation on the chosen tasks.

- **start_layer**  
  The layer index from which to begin merging.

- **result_path**  
  File path to save the evaluation results.

- **output_path**  
  Directory path to save the final merged model.

## 💡 MoE Model CLI Inference with vLLM
This project provides a **CLI-based inference script** for running Qwen3-30B-A3B (or a merged MoE variant) using `vLLM` \
It supports **thinking mode** (`<think>...</think>` blocks) and allows users to interactively query the model from the command line.


### ⚙️ Usage
#### 1. Run with Shell Script.
```bash
bash cli_infer.sh
```
- `--model_path`: Path to the model directory (Hugging Face format).

- `--tp_size`: Tensor parallel size (number of GPUs used).

- `--max_seqs`: Maximum number of sequences processed in parallel.
#### 2. Example Interaction
```bash
CLI Inference (Thinking Mode ON, type 'exit' to quit)

Prompt >>> What is mixture of experts in deep learning?

[Raw Model Output]
<think>Mixture of Experts (MoE) is a neural network architecture ...</think>
MoE is a technique that uses multiple experts (specialized sub-networks)
and a gating mechanism to route inputs to the most relevant experts.

[Thinking]
Mixture of Experts (MoE) is a way to improve efficiency by splitting
work among multiple sub-networks ...

[Answer]
MoE is a neural network design that routes inputs to specialized experts,
improving efficiency and scalability.
------------------------------------------------------------
```
- `Thinking` block shows the model’s hidden reasoning process.
- `Answer` block is the final user-facing response. 