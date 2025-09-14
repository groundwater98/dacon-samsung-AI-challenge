#!/bin/bash
MODEL_PATH="results/qwen/merge-96/test"
TP_SIZE=2
MAX_SEQS=512

python cli_infer.py \
  --model_path $MODEL_PATH \
  --tp_size $TP_SIZE \
  --max_seqs $MAX_SEQS