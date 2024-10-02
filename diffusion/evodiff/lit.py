# Some Code Adapted from Google's JAX Implementation: https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
from typing import Any, Dict, List

import pytorch_lightning as pl
import pydantic
from src.experimental.evodiff.model import ByteNetLMTime, BiMamba
from src.experimental.evodiff.losses import OAMaskedCrossEntropyLoss
from mamba_ssm.models.mixer_seq_simple import MambaConfig
from src.experimental.optimizers import build_optimizer_and_scheduler
from sequence_models.metrics import MaskedAccuracy
import math

class LitEvoDiffConfig(pydantic.BaseModel):

    # ============
    # Model Fields
    # ============

    hidden_features: int = 256
    num_layers: int = 8


    # ===============
    # Training Fields
    # ===============

    lr: float = 2e-3
    betas: List[float] = (0.9, 0.95)
    wd: float = 0.1

    scheduler_span: int = 50000

    padding_idx: int = 5


class LitEvoDiff(pl.LightningModule):


    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters()
        
        self.config: LitEvoDiffConfig = LitEvoDiffConfig.parse_obj(config)
        cfg = self.config

        self.evodiff = BiMamba(config=MambaConfig(
                d_model=cfg.hidden_features,
                n_layer=cfg.num_layers,
                vocab_size=(4 + 1),
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                pad_vocab_size_multiple=1,
            ))

        self.loss_func = OAMaskedCrossEntropyLoss(reweight=True)
        self.accu_func = MaskedAccuracy()

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
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def _step(self, batch, split):
     
        padding_idx = self.config.padding_idx
        src, timestep, tgt, mask = batch
        input_mask = (src != padding_idx).float()

        # n_tokens = mask.sum()

        # n_processed = input_mask.sum()
        # n_seqs = torch.tensor(len(src), device=src.device)
        outputs = self.evodiff(src, input_mask=input_mask)
        ce_loss, nll_loss = self.loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
        loss = ce_loss
        acc = self.accu_func(outputs, tgt, input_mask)
        sync = split != "train"
        self.log(f"{split}/loss", loss.item(), sync_dist=sync)
        self.log(f"{split}/nll", nll_loss.item(), sync_dist=sync)
        self.log(f"{split}/acc", acc.item(), sync_dist=sync)

        return loss