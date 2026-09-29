#!/bin/env python3

# Install dependencies
# pip install torch transformers datasets tensorboard

# Python libs
import time
import os
import argparse

# Libraries
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader

from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DatasetDict

# Additional scripts
import eval_model
import token_model


#### PARAMETERS ################################################################

MODEL_NAME = "google-bert/bert-base-uncased"

parser = argparse.ArgumentParser(description="Training BERT script")
parser.add_argument("-b", "--batch-size", type=int, default=16)
parser.add_argument("--max-length", type=int, default=384) # max 512
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("-c", "--workers", type=int, default=8)
parser.add_argument("-t", "--train-subset", type=int, default=None)
parser.add_argument("-v", "--validation-subset", type=int, default=None)
parser.add_argument("-l", "--log-steps", type=int, default=20)
parser.add_argument("-s", "--save-steps", type=int, default=200)

parser.add_argument("-p", "--profiler", type=str, default="./log/baseline")
parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pt")
parser.add_argument("--model", type=str, default="checkpoints/model.pt")
parser.add_argument("-r", "--resume", action="store_true", default=False)

args = parser.parse_args()

#### MODEL LOAD ################################################################

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit(1)

device = torch.device("cuda")
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


#### DATASET LOAD ##############################################################

squad = load_dataset("rajpurkar/squad")

split = squad["train"].train_test_split(test_size=0.1, seed=82749)
raw = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
    "test": squad["validation"],
})

if args.train_subset is not None:
    raw["train"] = raw["train"].select(range(min(args.train_subset, len(raw["train"]))))
if args.validation_subset is not None:
    raw["validation"] = raw["validation"].select(range(min(args.validation_subset, len(raw["validation"]))))

print(f"Train: {len(raw['train'])} examples | Validation: {len(raw['validation'])} examples")

def tokenize_train(examples):
    return token_model.tokenize_train(tokenizer, examples, max_length=args.max_length)
def tokenize_val(examples):
    return token_model.tokenize_val(tokenizer, examples, max_length=args.max_length)

train_tok = raw["train"].map(tokenize_train, batched=True, remove_columns=raw["train"].column_names)
train_tok.set_format("torch")
train_loader = DataLoader(
    train_tok,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.workers,
)

val_features = raw["validation"].map(tokenize_val, batched=True)
model_columns = ("input_ids", "attention_mask", "token_type_ids")
val_inputs = val_features.remove_columns([c for c in val_features.column_names if c not in model_columns])
val_inputs.set_format("torch")
val_loader = DataLoader(val_inputs, batch_size=args.batch_size, shuffle=False)


#### CHECKPOINTS ###############################################################

def save_checkpoint(epoch, step):
    tmp = args.checkpoint + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, tmp)
    os.replace(tmp, args.checkpoint)

total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

start_epoch = 0
global_step = 0

os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
if args.resume and os.path.exists(args.checkpoint):
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler(args.profiler),
    record_shapes=True,
) as prof:

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    overhead = 0.0

    for epoch in range(start_epoch, args.epochs):
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

            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}][step {global_step}/{total_steps}] loss={running_loss / args.log_steps:.4f}")
                running_loss = 0.0

            if global_step % args.save_steps == 0:
                save_checkpoint(epoch, global_step)

        eval_t0 = time.perf_counter()
        save_checkpoint(epoch + 1, global_step)
        em, f1 = eval_model.evaluate(device, model, val_loader, val_features)
        print(f"[epoch {epoch}] validation EM={em:.4f} F1={f1:.4f}")
        eval_t1 = time.perf_counter()
        overhead += eval_t1 - eval_t0

    torch.cuda.synchronize()
    t1 = time.perf_counter()


print(f"Total time: {t1 - t0:.4f}s\nOverhead: {overhead:.4f}s\nTraining time: {t1 - t0 - overhead:.4f}s")
torch.save(model.state_dict(), args.model)
print("Final model saved")

