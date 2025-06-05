from datasets import load_dataset
import os

# 1) pull the HaluEval “qa” split
raw = load_dataset("pminervini/HaluEval", "qa")

SYSTEM = (
  "A conversation between User and Assistant. The user asks a question, "
  "and the Assistant solves it. The assistant first thinks step-by-step "
  "and then provides the answer, as well as how confident they are in their answer."
  "Enclose reasoning in <think>…</think> "
  "and the final answer in <answer>…</answer>"
  "and the self confidence in <confidence>…</confidence>, where confidence is a number between 0.0 and 1.0."
)


def remap(example):
    prompt = [
        {"role":"system", "content":SYSTEM},
        {"role":"user",   "content":example["question"]},
    ]
    return {
        "prompt": prompt,
        "solution": example["right_answer"],
        "is_hallucinated": int(example["hallucinated_answer"] != example["right_answer"])
    }

# 2) apply remap and write out to data/halu_test/test
ds = raw["data"].map(remap, remove_columns=raw["data"].column_names)
os.makedirs("data/halu_test", exist_ok=True)
ds.save_to_disk("data/halu_test/test")
print(f"Wrote {len(ds)} examples to data/halu_test/test")

