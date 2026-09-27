# BERT-Base Fine-Tuning on SQuAD — Baseline Implementation

This project fine-tunes **BERT-base-uncased** ([google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)) for extractive Question Answering on the **SQuAD** dataset ([rajpurkar/squad](https://rajpurkar.github.io/SQuAD-explorer/)), using a single GPU. The implementation uses the Hugging Face `transformers` and `datasets` libraries on top of PyTorch.

The code is mainly organized into four files:

- `train.py` — main training script (data loading, training loop, checkpointing, timing, evaluation)
    - **Model and tokenizer loading**: loads `BertForQuestionAnswering` (BERT-base with a QA head that outputs a *start* and *end* logit per token) and a `BertTokenizerFast` (needed because it provides character-to-token offset mappings, required to convert character-based answer positions into token positions).
    - **Dataset loading and splitting**: loads SQuAD via `datasets.load_dataset`, then re-splits the original `train` split into a 90% train / 10% validation set, and reserves the official SQuAD `validation` split as a held-out `test` set.
    - **Tokenization**
    - **Checkpointing and resume**
    - **Training loop**
    - **Timing measurement**
    - **Profiling**
    - **Final evaluation**: prints **Exact Match (EM)** and **F1**.
- `token_model.py` — tokenization and answer-span label generation
- `eval_model.py` — SQuAD metrics (Exact Match / F1) and answer-span decoding
- `infer.py` — interactive command-line inference on a trained checkpoint

---

## Time and Profiling Report

### Training configuration used
ACTUALIZAR
| Parameter | Value |
|---|---|
| Model | `google-bert/bert-base-uncased` |
| Dataset | SQuAD (`rajpurkar/squad`) |
| Train examples used | `TRAIN_SUBSET_SIZE = 2000` |
| Validation examples used | `VAL_SUBSET_SIZE = 200` |
| Batch size | 16 |
| Max sequence length | 384 |
| Epochs | 1 |
| Learning rate | 2e-5 |
| GPU | A100 |

### Measured wall-clock training time

PONER RESULTADOS

### Validation results

PONER RESULTADOS

### Profiler output

Ponemos datos aqui o suprimimos?

---

## Dependencies
```
pip install tensorflow[and-cuda] torch transformers datasets tensorboard
```