#!/usr/bin/env python3
# src/train.py

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple, Union
import numpy as np
import math
import re

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ─── Hyperparameters & Defaults ─────────────────────────────────────────────
DEFAULT_LR             = 1e-5
DEFAULT_LORA_R         = 8
DEFAULT_LORA_ALPHA     = 32
DEFAULT_LORA_DROPOUT   = 0.1
DEFAULT_EPOCHS         = 1
DEFAULT_BATCH_SIZE     = 4
DEFAULT_GRAD_ACCUM     = 1
DEFAULT_MAX_PROMPT     = 128
DEFAULT_MAX_COMPLETION = 64
DEFAULT_NUM_GEN        = 4

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─── Utility: Sequence Statistics ─────────────────────────────────────────────
def compute_seq_stats(
    full_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> Tuple[float, float]:
    """
    Given prompt+generation, regenerate with output_scores=True,
    then compute:
      - average token log-prob
      - maximum token entropy (the “spike”)
    """
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    max_len = max(200, seq_len + 1)  # ensure max_length > input length
    generation = model.generate(
        **inputs,
        max_length=max_len,
        return_dict_in_generate=True,
        output_scores=True,
    )
    token_ids   = generation.sequences[0][1:].tolist()
    logits_list = generation.scores

    log_probs = []
    entropies = []
    zScores = []
    for tok_id, logits in zip(token_ids, logits_list):
        probs = torch.softmax(logits[0], dim=-1)
        logp  = torch.log(probs[tok_id] + 1e-20).item()
        ent   = -(probs * torch.log(probs + 1e-20)).sum().item()
        log_probs.append(logp)
        entropies.append(ent)
        if len(entropies) > 0:
            running_mu = np.mean(entropies)
            running_var = np.var(entropies)
            if abs(ent - running_mu) > math.sqrt(running_var):
                zScores.append(max(0, ent - running_mu))
            else:
                zScores.append(0)
        else:
            zScores.append(0)

    avg_logp  = np.mean(log_probs)
    max_spike = max(entropies)
    sum_zScores = -np.sum(zScores)
    return avg_logp, max_spike, sum_zScores

# ─── Reward Functions (closures capturing model & tokenizer) ────────────────
def make_logprob_reward(model, tokenizer):
    def logprob_reward(
        prompts: List[List[dict]],
        completions: List[Union[str, List[str], List[dict]]],
        **_
    ):
        rewards = []
        for prompt, gen in zip(prompts, completions):
            # normalize gen into one string
            if isinstance(gen, str):
                gen_text = gen
            elif isinstance(gen, list):
                if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                    gen_text = "".join(turn["content"] for turn in gen)
                else:
                    gen_text = "".join(gen)
            else:
                raise TypeError(f"Unexpected completion type: {type(gen)}")

            prompt_str = " ".join(turn["content"] for turn in prompt)
            avg_lp, _, _ = compute_seq_stats(prompt_str + gen_text, model, tokenizer)
            rewards.append(avg_lp)

        logger.info(f"  ▶ logprob_reward mean: {sum(rewards)/len(rewards):.3f}")
        return rewards
    return logprob_reward

def make_entropy_reward(model, tokenizer):
    def entropy_reward(
        prompts: List[List[dict]],
        completions: List[Union[str, List[str], List[dict]]],
        **_
    ):
        rewards = []
        for prompt, gen in zip(prompts, completions):
            # normalize gen into one string
            if isinstance(gen, str):
                gen_text = gen
            elif isinstance(gen, list):
                if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                    gen_text = "".join(turn["content"] for turn in gen)
                else:
                    gen_text = "".join(gen)
            else:
                raise TypeError(f"Unexpected completion type: {type(gen)}")

            prompt_str = " ".join(turn["content"] for turn in prompt)
            _, max_ent, _ = compute_seq_stats(prompt_str + gen_text, model, tokenizer)
            rewards.append(-max_ent)

        logger.info(f"  ▶ entropy_reward mean: {sum(rewards)/len(rewards):.3f}")
        return rewards
    return entropy_reward

def make_zScore_reward(model, tokenizer):
    def zScore_reward(
        prompts: List[List[dict]],
        completions: List[Union[str, List[str], List[dict]]],
        **_
    ):
        rewards = []
        for prompt, gen in zip(prompts, completions):
            # normalize gen into one string
            if isinstance(gen, str):
                gen_text = gen
            elif isinstance(gen, list):
                if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                    gen_text = "".join(turn["content"] for turn in gen)
                else:
                    gen_text = "".join(gen)
            else:
                raise TypeError(f"Unexpected completion type: {type(gen)}")

            prompt_str = " ".join(turn["content"] for turn in prompt)
            _, _, zScores = compute_seq_stats(prompt_str + gen_text, model, tokenizer)
            rewards.append(zScores)

        logger.info(f"  ▶ zScore_reward mean: {sum(rewards)/len(rewards):.3f}")
        return rewards
    return zScore_reward


def extract_confidence(text: str) -> float:
    """
    Extracts the confidence value from the <confidence> tag in the model's output.
    Returns a float ∈ [0.0, 1.0]. Defaults to 0.5 if not found or invalid.
    """
    match = re.search(r"<confidence>\s*(\d*\.?\d+)\s*</confidence>", text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))  # clip to [0, 1]
        except ValueError:
            pass
    return 0.5


def make_confidence_reward(model, tokenizer, hallucination_labels: List[int]):
    """
    Builds a reward function using confidence alignment.
    Reward = 2 * (confidence - 0.5) * y
    where y = -1 if hallucination is present, +1 if not
    """
    def confidence_reward(
        prompts: List[List[dict]],
        completions: List[Union[str, List[str], List[dict]]],
        **_
    ):
        rewards = []
        for i, (prompt, gen) in enumerate(zip(prompts, completions)):
            # normalize generation to string
            if isinstance(gen, str):
                gen_text = gen
            elif isinstance(gen, list):
                if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                    gen_text = "".join(turn["content"] for turn in gen)
                else:
                    gen_text = "".join(gen)
            else:
                raise TypeError(f"Unexpected completion type: {type(gen)}")

            # extract confidence from output
            confidence = extract_confidence(gen_text)

            # hallucination label from dataset: 1 = NOT hallucinated, -1 = hallucinated
            y = 1 if hallucination_labels[i] == 0 else -1

            # compute reward
            reward = 2 * (confidence - 0.5) * y
            rewards.append(reward)

        logger.info(f"  ▶ confidence_reward mean: {sum(rewards)/len(rewards):.3f}")
        return rewards

    return confidence_reward







# ─── Argument Parsing ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA + GRPO training with logprob + entropy‐spike rewards"
    )
    p.add_argument("--model-id",   required=True,
                   help="Base HF model ID, e.g. Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--data-dir",   required=True,
                   help="Directory containing `train/` split from data preprocessing")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", "--learning-rate",
                   dest="lr", type=float, default=DEFAULT_LR,
                   help="Learning rate")
    return p.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1) Load preprocessed train split
    train_path = os.path.join(args.data_dir, "train")
    logger.info(f"Loading dataset from {train_path} …")
    train_ds = load_from_disk(train_path)

    # 2) Load base model & apply LoRA
    logger.info(f"Loading base model ({args.model_id}) + applying LoRA…")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    lora_cfg  = LoraConfig(
        task_type="CAUSAL_LM",
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        target_modules=["q_proj","v_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # 3) Make our two reward functions
    logprob_reward = make_logprob_reward(model, tokenizer)
    entropy_reward = make_entropy_reward(model, tokenizer)
    zScore_reward = make_zScore_reward(model, tokenizer)

    hallucination_labels = [int(ex["is_hallucinated"]) for ex in train_ds]
    conf_reward = make_confidence_reward(model, tokenizer, hallucination_labels)


    # 4) Configure GRPO
    logger.info("Configuring GRPO trainer…")
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
        bf16=True,
        max_prompt_length=DEFAULT_MAX_PROMPT,
        max_completion_length=DEFAULT_MAX_COMPLETION,
        num_generations=DEFAULT_NUM_GEN,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        push_to_hub=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        # reward_funcs=[logprob_reward, entropy_reward]
        # reward_funcs=[zScore_reward]
        reward_funcs=[conf_reward]
    )

    # 5) Train!
    logger.info("Starting training…")
    t0 = time.time()
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Done in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

