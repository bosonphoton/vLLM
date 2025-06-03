#!/bin/bash
python src/data.py \
  --dataset AI-MO/NuminaMath-TIR \
  --split-train train[:5%] \
  --split-test  test[:5%] \
  --output-dir data
