export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="your-huggingface-home-path"

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 dacon_lmls/merging-qwen.py \
  --model_name="Qwen/Qwen3-30B-A3B" \
  --task="arc_challenge,openbookqa,rte,winogrande" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hierarchical" \
  --linkage="average" \
  --merge="freq" \
  --pruning_ratio=0.77 \
  --num_average_groups=96 \
  --n_sentences=64 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="results/results_qwen30BA3B.txt" \
  --output_path="results/qwen/merge/test" |& tee results/log_test
