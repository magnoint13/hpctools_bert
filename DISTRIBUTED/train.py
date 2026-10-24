#!/usr/bin/env python3

import torch
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.profilers import PyTorchProfiler

from datasets import DatasetDict, load_dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast, get_linear_schedule_with_warmup

import eval_model
import token_model

MODEL_NAME = "google-bert/bert-base-uncased"


#### MODELO ####################################################################

class SquadBert(L.LightningModule):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        lr: float = 2e-5,
        warmup_steps: int = 0,
        compile: str|None = None,
        tf32: bool = False,
        fused: bool = False,
    ):
        super().__init__()
        # save the arguments into self.hparams
        self.save_hyperparameters()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

    def setup(self, _stage):
        torch.set_float32_matmul_precision("high" if self.hparams.tf32 else "highest")
        if self.hparams.compile:
            self.model.compile(mode=self.hparams.compile)

    def training_step(self, batch, _batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss, batch_size=batch["input_ids"].size(0))
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, fused=self.hparams.fused)
        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


#### DATOS #####################################################################

class SquadData(L.LightningDataModule):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 16, # per device
        max_length: int = 384, # max is 512
        workers: int = 8,
        train_subset: int|None = None,
        validation_subset: int|None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _raw(self):
        h = self.hparams
        squad = load_dataset("rajpurkar/squad")
        split = squad["train"].train_test_split(test_size=0.1, seed=82749)  # igual en todos los ranks
        raw = DatasetDict({"train": split["train"], "validation": split["test"], "test": squad["validation"]})
        if h.train_subset is not None:
            raw["train"] = raw["train"].select(range(min(h.train_subset, len(raw["train"]))))
        if h.validation_subset is not None:
            raw["validation"] = raw["validation"].select(range(min(h.validation_subset, len(raw["validation"]))))
        return raw

    def _tokenize(self, raw):
        h = self.hparams
        tokenizer = BertTokenizerFast.from_pretrained(h.model_name)

        def tokenize_train(examples):
            return token_model.tokenize_train(tokenizer, examples, max_length=h.max_length)

        def tokenize_val(examples):
            return token_model.tokenize_val(tokenizer, examples, max_length=h.max_length)

        train_tok = raw["train"].map(tokenize_train, batched=True, remove_columns=raw["train"].column_names)
        train_tok.set_format("torch")

        val_features = raw["validation"].map(tokenize_val, batched=True)
        cols = ("input_ids", "attention_mask", "token_type_ids")
        val_inputs = val_features.remove_columns([c for c in val_features.column_names if c not in cols])
        val_inputs.set_format("torch")
        return train_tok, val_features, val_inputs

    def prepare_data(self):
        # one process downloads and tokenizes, which creates a cache for the rest
        raw = self._raw()
        print(f"Train: {len(raw['train'])} examples | Validation: {len(raw['validation'])} examples")
        self._tokenize(raw)

    def setup(self, _stage=None):
        # here the cache is used
        self.train_tok, self.val_features, self.val_inputs = self._tokenize(self._raw())

    def train_dataloader(self):
        # with multiple devices, Lightning would use a DistributedSampler
        return DataLoader(
            self.train_tok,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.workers,
            persistent_workers=True,   # exige workers >= 1
            prefetch_factor=4,
            drop_last=True,
        )

    def eval_dataloader(self):
        # note that the validation is done in SquadEval
        return DataLoader(
            self.val_inputs,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


#### CALLBACK ##################################################################

class SquadEval(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # eval EM/F1 at the end of every epoch
        pl_module.eval()
        with torch.no_grad():
            em, f1 = eval_model.evaluate(
                pl_module.device,
                pl_module.model,
                trainer.datamodule.eval_dataloader(),
                trainer.datamodule.val_features
            )
        pl_module.train()
        pl_module.print(f"[epoch {trainer.current_epoch}] validation EM={em:.4f} F1={f1:.4f}")
        pl_module.log_dict({"val_em": em, "val_f1": f1}, rank_zero_only=True)

    def on_train_end(self, _trainer, pl_module):
        # print used memory
        pl_module.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9} GB")


#### CLI #######################################################################

class Cli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_name", "data.model_name")
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.add_lightning_class_args(SquadEval, "squad_eval")
        parser.set_defaults({
            "checkpoint.dirpath": "checkpoints",
            "checkpoint.filename": "checkpoint",
            "checkpoint.every_n_train_steps": 200,
        })


if __name__ == "__main__":
    Cli(
        SquadBert,
        SquadData,
        trainer_defaults={
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 1,
            "precision": "32-true",
            "log_every_n_steps": 20,
            "enable_progress_bar": False,
            "profiler": "simple",
        },
    )
