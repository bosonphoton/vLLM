#!/usr/bin/env python3
import argparse, time, re, os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    confusion_matrix, precision_score, recall_score
)

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive evaluation of fine-tuned LLM + detectors")
    p.add_argument("--model-dir",   required=True, help="Path or HF repo ID of your fine-tuned adapter")
    p.add_argument("--base-model",  required=True, help="Base LLM repo ID (must match fine-tune)")
    p.add_argument("--data-dir",    default="data", help="Either a `test/` subfolder or dataset root")
    p.add_argument("--num-samples", type=int, default=100, help="How many examples to eval")
    p.add_argument("--plots-dir",   default="plots", help="Where to save histogram & curve PNGs")
    return p.parse_args()

def generate_and_score(prompt, model, tokenizer):
    # build input string
    text = " ".join(turn["content"] for turn in prompt)
    # tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # generation
    start = time.time()
    out = model.generate(
        **inputs,
        max_length=200,
        return_dict_in_generate=True,
        output_scores=True,
    )
    elapsed = time.time() - start
    seq = out.sequences[0]
    # decode
    generated = tokenizer.decode(seq, skip_special_tokens=True)
    gen_len   = seq.size(-1) - inputs.input_ids.size(-1)
    # per-token scores
    logps, ents, zScoreEnt = [], [], []
    running_ent_mean = 0
    for tid, logits in zip(seq[1:], out.scores):
        probs = torch.softmax(logits[0], dim=-1)
        logps.append(torch.log(probs[tid] + 1e-20).item())
        ents.append(-(probs * torch.log(probs + 1e-20)).sum().item())
        if len(ents) > 0:
            running_ent_mean = np.mean(ents)
            running_var = np.var(ents)
            if abs(ents[-1] - running_ent_mean) > math.sqrt(running_var):
                zScoreEnt.append(max(0, ents[-1] - running_ent_mean))
        else:
            running_ent_mean = ents[0]
            zScoreEnt.append(0)
    return generated, elapsed, gen_len, float(np.mean(logps)), float(np.mean(ents)), float(np.mean(zScoreEnt))

def find_best_threshold(scores, labels):
    prec, rec, thresh = precision_recall_curve(labels, scores)
    f1 = 2*prec*rec/(prec+rec+1e-20)
    ix = np.nanargmax(f1)
    return thresh[ix], f1[ix], prec[ix], rec[ix]

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    # 1) load model
    base  = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base, args.model_dir, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # 2) load data
    test_path = os.path.join(args.data_dir, "test")
    ds = load_from_disk(test_path) if os.path.isdir(test_path) else load_from_disk(args.data_dir)
    n = min(args.num_samples, len(ds))

    # 3) run eval
    times, lengths, logps, ents, zScoreEnt = [], [], [], [], []
    fmt_ok = corr_ok = 0
    labels = []

    for ex in ds.select(range(n)):
        out, t, L, lp, ent, zScoreEnt = generate_and_score(ex["prompt"], model, tokenizer)
        times.append(t); lengths.append(L); logps.append(lp); ents.append(ent); zScoreEnt.append(zScoreEnt)
        lab = int(ex.get("is_hallucinated", 0))
        labels.append(lab)
        # format check
        if re.match(r"^<think>.*?</think>\s*<answer>.*?</answer>$", out.strip(), flags=re.DOTALL):
            fmt_ok += 1
        # exact-match
        m = re.search(r"<answer>(.*?)</answer>", out, flags=re.DOTALL)
        ans = m.group(1).strip() if m else ""
        if ans == ex["solution"].strip():
            corr_ok += 1

    labels = np.array(labels)
    times, lengths = np.array(times), np.array(lengths)
    logps, ents   = np.array(logps), np.array(ents)
    zScoreEnt     = np.array(zScoreEnt)

    # 4) metrics
    fmt_pct  = fmt_ok/n
    corr_pct = corr_ok/n
    auroc_lp = roc_auc_score(labels, -logps) if len(np.unique(labels))>1 else np.nan
    auroc_ent= roc_auc_score(labels, ents)    if len(np.unique(labels))>1 else np.nan
    auroc_zScore= roc_auc_score(labels, zScoreEnt)    if len(np.unique(labels))>1 else np.nan

    # best-threshold F1 for each
    th_lp,  f1_lp,  p_lp,  r_lp  = find_best_threshold(-logps, labels)
    th_ent, f1_ent, p_ent, r_ent = find_best_threshold(ents, labels)
    th_zScore, f1_zScore, p_zScore, r_zScore = find_best_threshold(zScoreEnt, labels)
    # union detector
    pred_union = ((-logps>=th_lp) | (ents>=th_ent)).astype(int)
    f1_union  = f1_score(labels, pred_union)
    prec_u, rec_u = precision_score(labels, pred_union), recall_score(labels, pred_union)

    # 5) print summary
    print("\n## Evaluation Summary\n")
    print(f"Avg latency (s):        {times.mean():.3f} ± {times.std():.3f}")
    print(f"Avg tokens:             {lengths.mean():.1f} ± {lengths.std():.1f}\n")
    print(f"Avg token log-prob:     {logps.mean():.3f} ± {logps.std():.3f}")
    print(f"Avg token entropy:      {ents.mean():.3f} ± {ents.std():.3f}\n")
    print(f"AUROC (log-prob detector) : {auroc_lp:.3f}")
    print(f"AUROC (entropy detector)  : {auroc_ent:.3f}\n")
    print(f"Best-F1 log-prob @ {th_lp:.3f}: F1={f1_lp:.3f}, prec={p_lp:.3f}, rec={r_lp:.3f}")
    print(f"Best-F1 entropy  @ {th_ent:.3f}: F1={f1_ent:.3f}, prec={p_ent:.3f}, rec={r_ent:.3f}")
    print(f"Best-F1 zScore  @ {th_zScore:.3f}: F1={f1_zScore:.3f}, prec={p_zScore:.3f}, rec={r_zScore:.3f}")
    print(f"Union detector: F1={f1_union:.3f}, prec={prec_u:.3f}, rec={rec_u:.3f}\n")

    # 6) save histograms
    for arr, name, xlabel in [
        (times,   "latency",      "Latency (s)"),
        (lengths, "gen_tokens",   "Generated Tokens"),
        (logps,   "logprob",      "Avg log-prob"),
        (ents,    "entropy",      "Avg entropy"),
    ]:
        plt.figure(); plt.hist(arr, bins="auto"); plt.title(xlabel)
        plt.xlabel(xlabel); plt.ylabel("Count"); plt.tight_layout()
        out = os.path.join(args.plots_dir, f"{name}_hist.png")
        plt.savefig(out); plt.close()
        print(f"Saved histogram: {out}")

    # 7) ROC & PR curves
    # ROC
    from sklearn.metrics import roc_curve
    fpr_lp, tpr_lp, _ = roc_curve(labels, -logps)
    fpr_ent, tpr_ent,_= roc_curve(labels, ents)
    plt.figure()
    plt.plot(fpr_lp, tpr_lp, label=f"logprob (AUC={auroc_lp:.2f})")
    plt.plot(fpr_ent, tpr_ent, label=f"entropy (AUC={auroc_ent:.2f})")
    plt.plot([0,1],[0,1],"--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC Curve")
    plt.tight_layout()
    fn = os.path.join(args.plots_dir,"roc_curve.png"); plt.savefig(fn); plt.close()
    print(f"Saved ROC curve: {fn}")

    # PR
    plt.figure()
    prec_lp, rec_lp, _ = precision_recall_curve(labels, -logps)
    prec_ent,rec_ent,_= precision_recall_curve(labels, ents)
    auc_lp = auc(rec_lp, prec_lp)
    auc_ent= auc(rec_ent,prec_ent)
    plt.plot(rec_lp, prec_lp, label=f"logprob (AUC={auc_lp:.2f})")
    plt.plot(rec_ent,rec_ent, label=f"entropy (AUC={auc_ent:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("PR Curve")
    plt.tight_layout()
    fn = os.path.join(args.plots_dir,"pr_curve.png"); plt.savefig(fn); plt.close()
    print(f"Saved PR curve: {fn}")

if __name__ == "__main__":
    main()

