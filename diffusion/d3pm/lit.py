# Some Code Adapted from Google's JAX Implementation: https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.experimental.d3pm.d3pm import D3PMConfig, D3PM
from src.experimental.d3pm.models import BiMamba
from src.experimental.llm.lit import LitLLMConfig
from src.experimental.optimizers import build_optimizer_and_scheduler
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel



class LitD3PM(pl.LightningModule):


    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters()

        self.full_config = config
        self.config: D3PMConfig = D3PMConfig.parse_obj(config)
        cfg = self.config
        self.mamba_config: LitLLMConfig = LitLLMConfig.parse_obj(config)
        mamba_cfg = self.mamba_config

        self.d3pm = D3PM(cfg)

        self.denoiser = BiMamba(
            config=MambaConfig(
                d_model=mamba_cfg.hidden_features,
                n_layer=mamba_cfg.num_layers,
                vocab_size=(4 + 1),
                rms_norm=(mamba_cfg.norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=mamba_cfg.fused_add_norm,
                pad_vocab_size_multiple=1,
            ))

    def configure_optimizers(self):
        cfg = self.mamba_config
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
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def _step(self, batch, split):

        x_start, mask = batch
        d3pm = self.d3pm
        cfg = self.config
        t = torch.randint(cfg.num_states, (x_start.shape[0],), device=x_start.device)

        x_t = d3pm.q_sample(x_start, t)
        x_t = x_t * mask
        pred_x_start_logits = self.denoiser(x_t, t)
        pred_x_start_logits = pred_x_start_logits.view(pred_x_start_logits.shape[1], pred_x_start_logits.shape[2], -1)

        loss = (d3pm.v_bound_L_t(self.denoiser, x_start, x_t, t, mask).mean()
                + cfg.loss_lambda * F.cross_entropy(pred_x_start_logits, (x_start - 1 + mask).T, ignore_index=-1))
        self.log(f"{split}/loss", loss)
        # if split == 'val':
        #     vb_prior, vb, full_vb = d3pm.full_v_bound(self.denoiser, x_start, mask)
        #     self.log(f"{split}/vb_prior", vb_prior)
        #     self.log(f"{split}/vb", vb)
        #     self.log(f"{split}/full_vb", full_vb)

        return loss


