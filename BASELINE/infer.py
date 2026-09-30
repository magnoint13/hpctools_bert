#!/bin/env python3

import argparse
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
import token_model

DEFAULT_MODEL_PATH = "checkpoints/model.pt"
MAX_LENGTH = 384

def answer_question(question, context, model, tokenizer, device):
    inputs = tokenizer(
        question, context,
        truncation="only_second", max_length=MAX_LENGTH,
        return_tensors="pt", return_offsets_mapping=True,
    )
    offsets = inputs.pop("offset_mapping")[0].tolist()
    sequence_ids = inputs.sequence_ids(0)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    start_idx = output.start_logits[0].argmax().item()
    end_idx = output.end_logits[0].argmax().item()

    if end_idx < start_idx or sequence_ids[start_idx] != 1 or sequence_ids[end_idx] != 1:
        return "(no se ha encontrado una respuesta clara en el contexto)"

    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]
    return context[start_char:end_char]


def main():
    parser = argparse.ArgumentParser(description="Inferencia manual con BERT fine-tuned en SQuAD")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased").to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    print(f"Modelo cargado desde {args.model_path} ({device})")
    try:
        while True:
            context = input("\nContexto: ")
            while True:
                question = input("\nPregunta (o 'salir'): ")
                if question.strip().lower() in ("salir", "exit", "quit"):
                    break
                print("Respuesta:", answer_question(question, context, model, tokenizer, device))

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
