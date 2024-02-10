from typing import Any, Dict, List, Literal

import lightning.pytorch as pl
import pydantic
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

    lr: float = 1e-4
    betas: List[float] = (0.9, 0.95)
    wd: float = 0.1

    # ================
    # Sampling Fields
    # ================


class LitLLM(pl.LightningModule):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters()

        self.full_config = config
        self.config: LitLLMConfig = LitLLMConfig.parse_obj(config)
        cfg = self.config

        # A=0 C=1 G=2 T=3 EOS=4 SOS=5
        self.eos = 4
        self.sos = 5  # need this to be last (!)

        self.mamba = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.hidden_features,
                n_layer=cfg.num_layers,
                vocab_size=(4 + 2),
                rms_norm=(cfg.norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=cfg.fused_add_norm,
                pad_vocab_size_multiple=1,
            )
        )

    def configure_optimizers(self):
        cfg = self.config
        optimizer, scheduler = build_optimizer_and_scheduler(
            self,
            lr=cfg.lr,
            betas=cfg.betas,
            wd=cfg.wd,
            warmup=2000,
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

        # Convert padding into EOS tokens
        sequences = torch.where(mask, sequences, self.eos)

        # Create shifted versions of the sequence
        inputs = F.pad(sequences, (1, 0), value=self.sos)  # [sos, ...] (B L+1)
        target = F.pad(sequences, (0, 1), value=self.eos)  # [..., eos] (B L+1)
        mask = F.pad(mask, (1, 0), value=True)

        # Forward pass
        logits = self.mamba(inputs, num_last_tokens=1).logits  # (B L+1 6)
        logits = logits[..., :-1]  # don't compute loss on SOS
        logits = logits.mT  # (B 5 L+1)

        # Compute loss
        losses = F.cross_entropy(logits, target, reduction="none")  # (B L+1)
        losses = torch.where(mask, losses, 0)
        loss = losses.sum() / mask.float().sum()

        # Logging
        log_kwargs = dict(batch_size=sequences.shape[0], sync_dist=(split != "train"))
        self.log(f"{split}/loss", loss, **log_kwargs)

        return loss
