import math
from typing import List, Literal, Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel

from src.datasets import DNATokenizer
from src.experimental.optimizers import build_optimizer_and_scheduler


class LitLLM(pl.LightningModule):
    """
    A PyTorch Lightning module for a Language Model using Mamba architecture.
    """

    def __init__(
        self,
        tokenizer_path: str,
        hidden_features: int = 512,
        num_layers: int = 22,
        ssm_cfg: Dict[str, Any] = {"layer": "Mamba2", "d_state": 64},
        norm: Literal["rms", "layer"] = "layer",
        fused_add_norm: bool = False,
        lr: float = 4e-3,
        betas: List[float] = (0.9, 0.95),
        wd: float = 0.1,
        scheduler_shape: Literal["hump", "flat"] = "hump",
        scheduler_span: int = 100000,
        num_samples_per_epoch: int = 10,
        max_length: int = 2048,
        top_k: int = -1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        """
        Initialize the LitLLM module.

        Args:
            tokenizer_path (str): Path to the tokenizer file.
            hidden_features (int): Number of hidden features in the model.
            num_layers (int): Number of layers in the model.
            ssm_cfg (Dict[str, Any]): Configuration for the SSM layer.
            norm (Literal["rms", "layer"]): Type of normalization to use.
            fused_add_norm (bool): Whether to use fused add norm.
            lr (float): Learning rate.
            betas (List[float]): Adam optimizer betas.
            wd (float): Weight decay.
            scheduler_shape (Literal["hump", "flat"]): Shape of the learning rate scheduler.
            scheduler_span (int): Span of the scheduler in steps.
            num_samples_per_epoch (int): Number of samples to generate per epoch.
            max_length (int): Maximum length of generated sequences.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p sampling parameter.
            min_p (float): Minimum probability for sampling.
            temperature (float): Sampling temperature.
            repetition_penalty (float): Penalty for token repetition.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.tokenizer: DNATokenizer = DNATokenizer(tokenizer_path)
        self.mamba: MambaLMHeadModel = MambaLMHeadModel(
            config=MambaConfig(
                d_model=hidden_features,
                n_layer=num_layers,
                ssm_cfg=ssm_cfg,
                vocab_size=len(self.tokenizer),
                rms_norm=(norm == "rms"),
                residual_in_fp32=True,
                fused_add_norm=fused_add_norm,
                pad_vocab_size_multiple=1,
            )
        )

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Generate sequences using the Mamba model.

        Args:
            *args: Positional arguments to pass to the generate method.
            **kwargs: Keyword arguments to pass to the generate method.

        Returns:
            torch.Tensor: Generated sequences.
        """
        return self.mamba.generate(*args, **kwargs)

    def lr_schedule(self, step: int) -> float:
        """
        Calculate the learning rate for a given step.

        Args:
            step (int): Current step.

        Returns:
            float: Learning rate for the given step.
        """
        hp = self.hparams
        min_lr, max_lr = 1e-4, hp.lr
        T = hp.scheduler_span
        warmup = int(0.1 * T)

        if step <= warmup:
            return max_lr * (step / warmup)
        elif hp.scheduler_shape == "flat":
            return hp.lr
        elif warmup < step <= T:
            scale = 1 + math.cos(math.pi * ((step - warmup) / (T - warmup)))
            return min_lr + 0.5 * (max_lr - min_lr) * scale
        else:
            return min_lr

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Configuration dictionary for the optimizer and scheduler.
        """
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss for the training step.
        """
        return self._step(batch, split="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a validation step.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss for the validation step.
        """
        return self._step(batch, split="val")

    def on_validation_epoch_end(self) -> None:
        """
        Log generated samples at the end of each validation epoch.
        """
        if (self.logger is None) or (self.global_rank != 0):
            return
        self.logger.log_text(
            "samples",
            columns=["sequence", "length"],
            data=[[x, len(x.replace(" ", ""))] for x in self._sample()],
            step=self.current_epoch,
        )

    def _step(self, batch: torch.Tensor, split: str) -> torch.Tensor:
        """
        Perform a forward step and calculate the loss.

        Args:
            batch (torch.Tensor): Input batch.
            split (str): Split name ('train' or 'val').

        Returns:
            torch.Tensor: Calculated loss.
        """
        dnas, mask = batch
        # (... L+1) (... L+1)

        # Forward pass
        logits = self.mamba(dnas[..., :-1]).logits.mT  # (... C L)
        mask = mask[..., 1:]  # (... L)

        # Compute loss
        num_tokens = mask.int().sum()
        losses = F.cross_entropy(logits, dnas[..., 1:], reduction="none")  # (... L)
        losses = torch.where(mask, losses, 0)
        loss = losses.sum() / num_tokens.float()

        # Logging
        self.log(
            f"{split}/loss", loss, batch_size=num_tokens, sync_dist=(split != "train")
        )

        return loss

    @rank_zero_only
    def _sample(self) -> List[str]:
        """
        Generate samples using the model.

        Returns:
            List[str]: List of generated DNA sequences.
        """
        hp = self.hparams
        bos = torch.full(
            size=[hp.num_samples_per_epoch, 1],
            fill_value=self.tokenizer.bos_token_id,
            device=self.device,
        )
        samples = self.generate(
            input_ids=bos,
            max_length=hp.max_length,
            top_k=hp.top_k,
            top_p=hp.top_p,
            min_p=hp.min_p,
            temperature=hp.temperature,
            repetition_penalty=hp.repetition_penalty,
            vocab_size=len(self.tokenizer),
        )
        return [self.tokenizer.decode_dna(x) for x in samples]
