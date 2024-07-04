import math
from typing import List, Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from torchmetrics import MeanMetric

from src.experimental.optimizers import build_optimizer_and_scheduler
from src.utils import TOKENIZER


class LitLLM(pl.LightningModule):

    def __init__(
        self,
        hidden_features: int = 512,
        Lmax: int = 2048,
        num_layers: int = 22,
        norm: Literal["rms", "layer"] = "layer",
        fused_add_norm: bool = False,
        lr: float = 4e-3,
        betas: List[float] = (0.9, 0.95),
        wd: float = 0.1,
        scheduler_shape: Literal["hump", "flat"] = "hump",
        scheduler_span: int = 100000,
        num_samples_per_epoch: int = 20,
        top_k: int = -1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.mamba = MambaLMHeadModel(
            config=MambaConfig(
                d_model=hidden_features,
                n_layer=num_layers,
                vocab_size=TOKENIZER.vocab_size,
                rms_norm=(norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=fused_add_norm,
                pad_vocab_size_multiple=1,
            )
        )

        self.metrics = nn.ModuleDict({"train_": MeanMetric(), "val_": MeanMetric()})

    def generate(self, *args, **kwargs):
        return self.mamba.generate(*args, **kwargs)

    def lr_schedule(self, step):
        hp = self.hparams
        min_lr, max_lr = 1e-4, hp.lr
        T = hp.scheduler_span
        warmup = int(0.1 * T)

        if hp.scheduler_span == "flat":
            return hp.lr
        elif step <= warmup:
            return max_lr * (step / warmup)
        elif warmup < step <= T:
            scale = 1 + math.cos(math.pi * ((step - warmup) / (T - warmup)))
            return min_lr + 0.5 * (max_lr - min_lr) * scale
        else:
            return min_lr

    def configure_optimizers(self):
        hp = self.hparams
        optimizer, scheduler = build_optimizer_and_scheduler(
            self,
            lr=self.lr_schedule,
            betas=hp.betas,
            wd=hp.wd,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, split="val")

    def on_validation_epoch_end(self) -> None:
        if (self.logger is None) or (self.global_rank != 0):
            return
        self.logger.log_text(
            "samples",
            columns=["sequence", "length"],
            data=[[x, len(x.replace(" ", ""))] for x in self._sample()],
            step=self.current_epoch,
        )

    def _step(self, batch, split):
        dnas, mask, finetune = batch
        # (B L+1) (B L+1) (B)

        # Forward pass
        logits = self.mamba(dnas[..., :-1]).logits.mT  # (B C L)
        mask = mask[..., 1:]  # (B L)

        # Compute loss
        num_tokens = mask.int().sum()
        losses = F.cross_entropy(logits, dnas[..., 1:], reduction="none")  # (B L)
        losses = torch.where(mask, losses, 0)
        loss = losses.sum() / num_tokens.float()

        # Subset to finetuning plasmids
        with torch.no_grad():
            mask_finetune = finetune.unsqueeze(-1) & mask
            loss_finetune = self.metrics[f"{split}_"]
            loss_finetune.update(losses, mask_finetune.float())

        # Logging
        self.log(f"{split}/loss", loss, batch_size=num_tokens, sync_dist=(split != "train"))
        self.log(f"{split}/loss_finetune", loss_finetune, on_step=False, on_epoch=True)

        return loss

    @rank_zero_only
    def _sample(self):
        hp = self.hparams
        sos = TOKENIZER.vocab[TOKENIZER.cls_token]
        sos = torch.full([hp.num_samples_per_epoch, 1], sos, device=self.device)
        samples = self.generate(
            input_ids=sos,
            max_length=hp.Lmax,
            top_k=hp.top_k,
            top_p=hp.top_p,
            min_p=hp.min_p,
            temperature=hp.temperature,
            repetition_penalty=hp.repetition_penalty,
            vocab_size=TOKENIZER.vocab_size,
        )
        samples = samples[..., 1:]  # remove prompt
        return [TOKENIZER.decode(x) for x in samples]
