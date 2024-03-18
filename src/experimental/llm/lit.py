import math
from typing import Any, Dict, List, Literal

import pydantic
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from torchmetrics import MeanMetric

from src.experimental.optimizers import build_optimizer_and_scheduler
from src.utils import PlasmidTokenizer


class LitLLMConfig(pydantic.BaseModel):

    # ============
    # Model Fields
    # ============

    hidden_features: int = 256
    num_layers: int = 16

    norm: Literal["rms", "layer"] = "layer"
    fused_add_norm: bool = False

    # ===============
    # Training Fields
    # ===============

    lr: float = 4e-3
    betas: List[float] = (0.9, 0.95)
    wd: float = 0.1

    scheduler_span: int = 100000


class LitLLM(pl.LightningModule):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters()

        self.full_config = config
        self.config: LitLLMConfig = LitLLMConfig.parse_obj(config)
        cfg = self.config

        self.tokenizer = PlasmidTokenizer()
        self.mamba = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.hidden_features,
                n_layer=cfg.num_layers,
                vocab_size=self.tokenizer.vocab_size,
                rms_norm=(cfg.norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=cfg.fused_add_norm,
                pad_vocab_size_multiple=1,
            )
        )

        self.metrics = nn.ModuleDict({"train_": MeanMetric(), "val_": MeanMetric()})

    def generate(self, *args, **kwargs):
        return self.mamba.generate(*args, **kwargs)

    def lr_schedule(self, step):
        cfg = self.config
        min_lr, max_lr = 1e-4, cfg.lr
        T = cfg.scheduler_span
        warmup = int(0.1 * T)

        if step <= warmup:
            return max_lr * (step / warmup)
        elif warmup < step <= T:
            scale = 1 + math.cos(math.pi * ((step - warmup) / (T - warmup)))
            return min_lr + 0.5 * (max_lr - min_lr) * scale
        else:
            return min_lr

    def configure_optimizers(self):
        cfg = self.config
        optimizer, scheduler = build_optimizer_and_scheduler(
            self,
            lr=self.lr_schedule,
            betas=cfg.betas,
            wd=cfg.wd,
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
