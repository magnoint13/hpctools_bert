import re
import string
import torch
from collections import Counter


def normalize_answer(s):
    """minúsculas, sin puntuación, sin artículos, sin espacios redundantes"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold, pred):
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def get_best_answer(start_logits, end_logits, offsets, context, max_answer_length=30):
    """Búsqueda simplificada del mejor span (start, end): recorre combinaciones válidas
    de índices con offset conocido y longitud razonable, maximizando la suma de logits.
    (Es una versión simplificada del post-procesado oficial de SQuAD, que hace una
    búsqueda n-best más cuidadosa; suficiente para este baseline.)"""
    start_logits = start_logits.detach().cpu().tolist()
    end_logits = end_logits.detach().cpu().tolist()
    valid_idx = [i for i, o in enumerate(offsets) if o is not None]
    if not valid_idx:
        return ""

    best_score, best_span = -1e9, (valid_idx[0], valid_idx[0])
    for s in valid_idx:
        for e in range(s, min(s + max_answer_length, valid_idx[-1] + 1)):
            if offsets[e] is None:
                continue
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best_score, best_span = score, (s, e)

    s, e = best_span
    return context[offsets[s][0]:offsets[e][1]]


@torch.no_grad()
def evaluate(device, model, val_loader, val_features):
    model.eval()
    exact_scores, f1_scores = [], []

    idx = 0
    for batch in val_loader:
        n = len(batch["input_ids"])
        inputs = {k: v.to(device) for k, v in batch.items()}
        output = model(**inputs)

        for i in range(n):
            example = val_features[idx + i]
            pred = get_best_answer(
                output.start_logits[i], output.end_logits[i],
                example["offset_mapping"], example["context"],
            )
            gold_answers = example["answers"]["text"]
            exact_scores.append(max(compute_exact(g, pred) for g in gold_answers))
            f1_scores.append(max(compute_f1(g, pred) for g in gold_answers))
        idx += n

    model.train()
    return sum(exact_scores) / len(exact_scores), sum(f1_scores) / len(f1_scores)
