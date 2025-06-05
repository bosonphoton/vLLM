#!/bin/bash
python src/train.py \
  --model-id Qwen/Qwen2-0.5B-Instruct \
  --data-dir data/halu_split \
  --output-dir outputs/halu_testing \
  --epochs 1 \
  --batch-size 4 \
  --lr 1e-5
