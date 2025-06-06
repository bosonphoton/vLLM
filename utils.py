import re
import math
from typing import List, Dict, Tuple


def extract_reasoning_and_answer(output) -> Tuple[str, str]:
    """
    Extracts the <think>...<think> reasoning section and final answer from a vLLM output object.
    """
    text = output[0].outputs[0].text

    # Extract reasoning from <think>...</think>
    reasoning_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    # Remove <think> section to isolate final answer
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Extract final sentence as the answer
    sentences = re.split(r'(?<=[.?!])\s+', cleaned_text)
    final_answer = next((s.strip() for s in reversed(sentences) if s.strip()), None)

    return reasoning, final_answer


def token_entropy(logprob_dict: Dict[int, object]) -> float:
    """
    Computes entropy from a single token's top-k logprobs.
    """
    probs = [math.exp(lp.logprob) for lp in logprob_dict.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def extract_sentences_and_logprobs(tokens: List[str], logprobs: List[Dict[int, object]]) -> List[Tuple[str, List[Dict[int, object]]]]:
    """
    Groups tokens and their logprobs into sentences based on punctuation.
    """
    sentences = []
    sentence_tokens = []
    sentence_logprobs = []

    for token, logprob in zip(tokens, logprobs):
        sentence_tokens.append(token)
        sentence_logprobs.append(logprob)

        if token in [".", "?", "!", "\n"]:
            sentence_text = "".join(sentence_tokens).strip()
            if sentence_text:
                sentences.append((sentence_text, sentence_logprobs))
            sentence_tokens = []
            sentence_logprobs = []

    # Add any remaining partial sentence
    if sentence_tokens:
        sentence_text = "".join(sentence_tokens).strip()
        if sentence_text:
            sentences.append((sentence_text, sentence_logprobs))

    return sentences


def sentence_entropy_scores(output_obj) -> List[Tuple[str, float]]:
    """
    Calculates average entropy per sentence from a vLLM output object.
    """
    logprobs_list = output_obj.logprobs  # List[Dict[int, Logprob]]

    # Reconstruct token strings using the highest logprob token at each step
    tokens = [
        max(lp_dict.items(), key=lambda kv: kv[1].logprob)[1].decoded_token
        for lp_dict in logprobs_list
    ]


    # Group token-logprob pairs into sentences
    sentence_chunks = extract_sentences_and_logprobs(tokens, logprobs_list)

    results = []
    for sentence, logprobs in sentence_chunks:
        entropies = [token_entropy(lp) for lp in logprobs]
        # avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        avg_entropy = max(entropies)
        results.append((sentence, avg_entropy))

    return results