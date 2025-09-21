#!/bin/bash
MODEL_PATH="results/qwen/merge/test"
TP_SIZE=2
MAX_SEQS=1024

python cli_infer.py \
  --model_path $MODEL_PATH \
  --tp_size $TP_SIZE \
  --max_seqs $MAX_SEQS
