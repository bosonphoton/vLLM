#!/bin/bash
python src/eval.py \
  --model-dir bkhalil1/HaluEval_vLLM \
  --base-model Qwen/Qwen2-0.5B-Instruct \
  --data-dir data \
  --num-samples $1
