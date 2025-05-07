from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
import re
from rouge_score import rouge_scorer

# Load the HaluEval dataset
dataset = load_dataset("pminervini/HaluEval", 'qa')
knowledge = dataset["data"]["knowledge"]
question = dataset["data"]["question"]
correct_answer = dataset["data"]["right_answer"]
hallucinated_answer = dataset["data"]["hallucinated_answer"]


# ----------------------------
# 1. Format Qwen prompts
# ----------------------------
def format_prompt(q):
    prompt = "Give the final answer wrapped in <<>>. For instance <<salt>>"
    return (
        "<|im_start|>system\nYou are a reasoning agent.<|im_end|>\n"
        f"<|im_start|>user\n{q + prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ----------------------------
# 2. Extract text inside << >>
# ----------------------------
def extract_answer(text):
    match = re.search(r"<<(.*?)>>", text)
    return match.group(1).strip() if match else text.strip()

def rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rouge1'].fmeasure

# ----------------------------
# 3. Load model with vLLM
# ----------------------------
llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=512, dtype="half")

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024
)


X = question[:10]
y = correct_answer[:10]

# Format prompts
prompts = [format_prompt(q) for q in X]

# ----------------------------
# 5. Generate predictions
# ----------------------------
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# ----------------------------
# 6. Compare predictions to ground truth
# ----------------------------

print("\n--- Results ---\n")
final_accuracy = 0
for i, output in enumerate(outputs):
    raw = output.outputs[0].text
    y_pred = extract_answer(raw)
    gt = y[i]
    rouge_f1 = rouge(gt,y_pred)

    print(f"Question: {X[i]}\n")
    print(f"Reasoning: {raw}\n") 
    print(f"Ground Truth: {gt}\n")
    print(f"AI Output: {y_pred}\n")
    print(f"Rouge: {rouge_f1}\n")
    
    print("---------------------------")
    
    final_accuracy += rouge_f1


print("FINAL ACCURACY ", final_accuracy / len(X))

