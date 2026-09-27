#!/bin/env python3

# Python libs
import time
import os

# Libraries
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader

from transformers import (
    BertTokenizer,
    BertModel,
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DatasetDict

# Additional scripts
import eval_model
import infer
import token_model


#### PARAMETERS ################################################################

BATCH_SIZE = 16
MAX_LENGTH = 384
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
MODEL_NAME = "google-bert/bert-base-uncased"
NUM_WORKERS = 8 # dataset loader

TRAIN_SUBSET_SIZE = 2000 # or None to use the whole dataset
VAL_SUBSET_SIZE = 200

LOG_EVERY_STEPS = 20
SAVE_EVERY_STEPS = 200
CHECKPOINT_PATH = "checkpoints/checkpoint.pt"
FINAL_MODEL_PATH = "checkpoints/model.pt"
RESUME = True


#### MODEL LOAD ################################################################

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit(1)

device = torch.device("cuda")
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


#### DATASET LOAD ##############################################################

squad = load_dataset("rajpurkar/squad")

split = squad["train"].train_test_split(test_size=0.1, seed=82749)
raw = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
    "test": squad["validation"],
})

if TRAIN_SUBSET_SIZE is not None:
    raw["train"] = raw["train"].select(range(min(TRAIN_SUBSET_SIZE, len(raw["train"]))))
if VAL_SUBSET_SIZE is not None:
    raw["validation"] = raw["validation"].select(range(min(VAL_SUBSET_SIZE, len(raw["validation"]))))

print(f"Train: {len(raw['train'])} examples | Validation: {len(raw['validation'])} examples")

def tokenize_train(examples):
    return token_model.tokenize_train(tokenizer, examples, max_length=MAX_LENGTH)
def tokenize_val(examples):
    return token_model.tokenize_val(tokenizer, examples, max_length=MAX_LENGTH)

train_tok = raw["train"].map(tokenize_train, batched=True, remove_columns=raw["train"].column_names)
train_tok.set_format("torch")
train_loader = DataLoader(
    train_tok,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)

val_features = raw["validation"].map(tokenize_val, batched=True)
model_columns = ("input_ids", "attention_mask", "token_type_ids")
val_inputs = val_features.remove_columns([c for c in val_features.column_names if c not in model_columns])
val_inputs.set_format("torch")
val_loader = DataLoader(val_inputs, batch_size=BATCH_SIZE, shuffle=False)


#### CHECKPOINTS ###############################################################

def save_checkpoint(epoch, step):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, CHECKPOINT_PATH)

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

start_epoch = 0
global_step = 0

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
if RESUME and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["step"]
    print(f"Starting from epoch {start_epoch} & step {global_step}")


#### TRAINING LOOP #############################################################

model.train()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log/baseline"),
    record_shapes=True,
) as prof:

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += output.loss.item()
            global_step += 1
            prof.step()

            if global_step % LOG_EVERY_STEPS == 0:
                print(f"[epoch {epoch}][step {global_step}/{total_steps}] loss={running_loss / LOG_EVERY_STEPS:.4f}")
                running_loss = 0.0

            if global_step % SAVE_EVERY_STEPS == 0:
                save_checkpoint(epoch, global_step)

    save_checkpoint(epoch + 1, global_step)
    em, f1 = eval_model.evaluate(device, model, val_loader, val_features)
    print(f"[epoch {epoch}] validation EM={em:.4f} F1={f1:.4f}")

    torch.cuda.synchronize()
    t1 = time.perf_counter()

print(f"Training time: {t1 - t0:.2f}s")
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("Final model saved")

