#!/bin/env python3

# Install dependencies
# pip install torch transformers datasets tensorboard

# Python libs
import time
import os
import argparse
import contextlib # disable AMP

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

# general config options
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

# file options
parser.add_argument("-p", "--profiler", type=str, default="./log/baseline")
parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pt")
parser.add_argument("--model", type=str, default="checkpoints/model.pt")
parser.add_argument("-r", "--resume", action="store_true", default=False)

# optimization options
parser.add_argument("--compile", type=str, choices=("default", "reduce-overhead", "max-autotune"), default=None)
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument("--tf32", action="store_true", default=False)
parser.add_argument("--fused", action="store_true", default=False)

args = parser.parse_args()

#### MODEL LOAD ################################################################

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit(1)

# lower precision
torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

device = torch.device("cuda")
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

if args.compile:
    print("Model compilation started")
    t0 = time.perf_counter()
    model = torch.compile(model, mode=args.compile)
    t1 = time.perf_counter()
    print(f"Model compilation finished in {t1 - t0:.4f}s")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=args.fused)


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
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)

val_features = raw["validation"].map(tokenize_val, batched=True)
model_columns = ("input_ids", "attention_mask", "token_type_ids")
val_inputs = val_features.remove_columns([c for c in val_features.column_names if c not in model_columns])
val_inputs.set_format("torch")
val_loader = DataLoader(
    val_inputs,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)


#### CHECKPOINTS ###############################################################

# compiled models store their keys with the preffix '_orig_mod.'
def unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def save_checkpoint(epoch, step):
    print("Model checkpoint started")
    t0 = time.perf_counter()

    tmp = args.checkpoint + ".tmp"
    torch.save({
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, tmp)
    os.replace(tmp, args.checkpoint)

    t1 = time.perf_counter()
    print(f"Model checkpoint finished in {t1 - t0:.4f}s")

total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

start_epoch = 0
global_step = 0

os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
if args.resume and os.path.exists(args.checkpoint):
    print("Model resume started")
    t0 = time.perf_counter()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    unwrap(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["step"]
    print(f"Starting from epoch {start_epoch} & step {global_step}")

    t1 = time.perf_counter()
    print(f"Model resume finished in {t1 - t0:.4f}s")


#### TRAINING LOOP #############################################################

amp_ctx = (
    torch.autocast("cuda", dtype=torch.bfloat16)
    if args.amp else
    contextlib.nullcontext()
)

model.train()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler(args.profiler),
    record_shapes=True,
) as prof:

    eval_overhead = 0.0
    checkpoint_overhead = 0.0

    torch.cuda.synchronize()
    total_t0 = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        running_loss = torch.zeros((), device=device)

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                output = model(**batch)
            output.loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += output.loss.detach()
            global_step += 1
            prof.step()

            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}][step {global_step}/{total_steps}] loss={running_loss.item() / args.log_steps:.4f}")
                running_loss = torch.zeros((), device=device)

            if global_step % args.save_steps == 0:
                checkpoint_t0 = time.perf_counter()
                save_checkpoint(epoch, global_step)
                checkpoint_t1 = time.perf_counter()
                checkpoint_overhead += checkpoint_t1 - checkpoint_t0

        eval_t0 = time.perf_counter()
        save_checkpoint(epoch + 1, global_step)
        em, f1 = eval_model.evaluate(device, model, val_loader, val_features)
        print(f"[epoch {epoch}] validation EM={em:.4f} F1={f1:.4f}")
        eval_t1 = time.perf_counter()
        eval_overhead += eval_t1 - eval_t0

    torch.cuda.synchronize()
    total_t1 = time.perf_counter()

    total_overhead = checkpoint_overhead + eval_overhead
    total_elapsed_time = t1 - t0
    training_time = total_elapsed_time - total_overhead
    print(f"""\
Total time:          {total_elapsed_time:.4f} s
Checkpoint overhead: {checkpoint_overhead:.4f} s
Eval overhead:       {eval_overhead:.4f} s
Total overhead:      {total_overhead:.4f} s
Training time:       {training_time:.4f} s""")

    save_t0 = time.perf_counter()
    torch.save(model.state_dict(), args.model)
    save_t1 = time.perf_counter()
    print(f"Final model saved in {save_t1 - save_t0:.4f} s")

    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9} GB")
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace(args.profiler + "_chrome.trace")

