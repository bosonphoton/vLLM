from datasets import load_from_disk

def sanity_check_prompts(ds, split_name):
    bad = []
    for i, ex in enumerate(ds):
        # 1) prompt must be a list of two messages
        if not isinstance(ex["prompt"], list) or len(ex["prompt"]) != 2:
            bad.append((i, "wrong prompt structure"))
            continue

        sys_msg, user_msg = ex["prompt"]

        # 2) system message should contain <think>… and mention <answer> tags in guidance
        if "<think>" not in sys_msg["content"] or "<answer>" not in sys_msg["content"]:
            bad.append((i, "system prompt missing <think> or <answer> guidance"))

        # 3) user message should be clean (no tags yet)
        if "<think>" in user_msg["content"] or "<answer>" in user_msg["content"]:
            bad.append((i, "user prompt already has tags"))

        # 4) solution field must exist and be non-empty
        if "solution" not in ex or not ex["solution"].strip():
            bad.append((i, "missing or empty solution"))

    if bad:
        print(f"\n⚠️  Found {len(bad)} issues in {split_name} split:")
        for idx, msg in bad[:10]:
            print(f"  • #{idx}: {msg}")
        raise RuntimeError(f"{split_name} sanity check failed")
    print(f"✅ All {len(ds)} examples in {split_name} look good.")


def verify_counts(ds, expected_count, split_name):
    actual = len(ds)
    print(f"{split_name:>5}: {actual} examples (expected {expected_count})")
    if expected_count is not None and actual != expected_count:
        raise RuntimeError(f"{split_name} has {actual} examples but expected {expected_count}")


def main():
    # adjust these if we ever need change our splits
    expected = {"train": 3622, "test": 5}

    for split in ["train", "test"]:
        path = f"data/{split}"
        ds = load_from_disk(path)
        verify_counts(ds, expected[split], split)
        sanity_check_prompts(ds, split)

    print("\n🎉 All data splits passed verification!\n")

    # Show the first 5 test examples
    print("Here are your 5 test examples:\n")
    test_ds = load_from_disk("data/test")
    for i, ex in enumerate(test_ds):
        sys_msg, user_msg = ex["prompt"]
        print(f"--- example {i} ---")
        print("system:", sys_msg["content"])
        print("user:  ", user_msg["content"])
        print()
        if i >= 4:
            break


if __name__ == "__main__":
    main()

