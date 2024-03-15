import math
from typing import Any, Dict, List, Literal

import pydantic
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel

from src.experimental.optimizers import build_optimizer_and_scheduler


class LitLLMConfig(pydantic.BaseModel):

    # ============
    # Model Fields
    # ============

    hidden_features: int = 256
    num_layers: int = 16

    norm: Literal["rms", "layer"] = "rms"
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

        # A=0 C=1 G=2 T=3 EOS=4
        self.eos = 4

        self.mamba = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.hidden_features,
                n_layer=cfg.num_layers,
                vocab_size=(4 + 1),
                rms_norm=(cfg.norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=cfg.fused_add_norm,
                pad_vocab_size_multiple=1,
            )
        )

    def generate(self, *args, **kwargs):
        return self.mamba.generate(*args, **kwargs)

    def lr_schedule(self, step):
        cfg = self.config
        min_lr, max_lr = 1e-5, cfg.lr
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
        sequences, mask = batch  # (B L) (B L)

        sequences = torch.where(mask, sequences, self.eos)
        sequences = F.pad(sequences, (1, 1), value=self.eos)  # [eos, ..., eos] (B L+2)
        mask = F.pad(mask, (1, 0), value=True)  # [True] + mask

        # Forward pass
        logits = self.mamba(sequences[..., :-1]).logits.mT  # (B 5 L+1)

        # Compute loss
        losses = F.cross_entropy(logits, sequences[..., 1:], reduction="none")  # (B L+1)
        losses = torch.where(mask, losses, 0)
        loss = losses.sum() / mask.float().sum()

        # Logging
        log_kwargs = dict(batch_size=sequences.shape[0], sync_dist=(split != "train"))
        self.log(f"{split}/loss", loss, **log_kwargs)

        return loss
