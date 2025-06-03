#!/bin/bash
python src/train.py \
  --model-id Qwen/Qwen2-0.5B-Instruct \
  --data-dir data \
  --output-dir outputs \
  --epochs 1 \
  --batch-size 4 \
  --lr 1e-5
