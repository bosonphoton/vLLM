import argparse
from datasets import load_dataset

SYSTEM_PROMPT = (
  "A conversation between User and Assistant. The user asks a question, "
  "and the Assistant solves it. The assistant first thinks step-by-step "
  "and then provides the answer, as well as how confident they are in their answer."
  "Enclose reasoning in <think>…</think> "
  "and the final answer in <answer>…</answer>"
  "and the self confidence in <confidence>…</confidence>, where confidence is a number between 0.0 and 1.0."
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["problem"]},
        ],
        "solution": example["solution"],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     type=str, required=True)
    p.add_argument("--split-train", type=str, default="train[:5%]")
    p.add_argument("--split-test",  type=str, default="test[:5%]")
    p.add_argument("--output-dir",  type=str, required=True)
    args = p.parse_args()

    train_ds, test_ds = load_dataset(
        args.dataset,
        split=[ args.split_train, args.split_test ]
    )
    train_ds = train_ds.map(make_conversation, remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(make_conversation,  remove_columns=test_ds.column_names)

    train_ds.save_to_disk(f"{args.output_dir}/train")
    test_ds.save_to_disk( f"{args.output_dir}/test")

if __name__ == "__main__":
    main()

