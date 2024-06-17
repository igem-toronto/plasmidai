import pathlib
from typing import Literal, Optional

import pydantic_cli
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datasets import PlasmidDataModule
from src.experimental.callbacks import GradNormCallback
from src.experimental.llm.lit import LitLLM, LitLLMConfig
from src.paths import LOG_DIR, random_checkpoint_dir


class TrainLLMConfig(LitLLMConfig):

    seed: int = 100

    accelerator: str = "cpu"
    devices: int = 1
    strategy: Optional[str] = "auto"

    matmul_precision: Literal["medium", "high", "highest"] = "highest"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"

    # =================
    # Datamodule Fields
    # =================

    batch_size: int = 64
    num_workers: int = 8

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = -1
    train_steps_per_epoch: int = 50
    val_steps_per_epoch: int = 50

    finetune_path: Optional[str] = None
    resume_path: Optional[str] = None

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_project: str = "train_plasmid_llm"
    wandb_entity: Optional[str] = None
    wandb_dir: Optional[str] = None

    checkpoint: bool = False
    checkpoint_dir: Optional[str] = None

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainLLMConfig):
    cfg = config

    # Torch settings
    if cfg.accelerator == "gpu":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    data = PlasmidDataModule(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        finetune=(cfg.finetune_path is not None),
    )

    # Initialize trainer
    callbacks = [
        ModelSummary(max_depth=2),
        GradNormCallback(),
    ]

    if cfg.wandb:
        if cfg.wandb_dir is None:
            cfg.wandb_dir = LOG_DIR
        pathlib.Path(cfg.wandb_dir).mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=cfg.wandb_dir,
        )
        callbacks.append(LearningRateMonitor())
    else:
        logger = False

    if cfg.checkpoint:
        if cfg.checkpoint_dir is None:  # set to some random unique folder
            cfg.checkpoint_dir = random_checkpoint_dir()
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                filename="epoch={epoch}-loss={val/loss_finetune:.3f}",
                auto_insert_metric_name=False,
                monitor="val/loss_finetune",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        strategy=cfg.strategy,
        callbacks=callbacks,
        enable_checkpointing=cfg.checkpoint,
        logger=logger,
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.train_steps_per_epoch,
        limit_val_batches=cfg.val_steps_per_epoch,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.progress_bar,
        use_distributed_sampler=True,
    )

    # Initialize and load model
    if cfg.finetune_path is not None:
        llm = LitLLM.load_from_checkpoint(
            cfg.finetune_path,
            map_location="cpu",
            config=dict(cfg),
        )
        llm.requires_grad_(False)
        llm.mamba.backbone.layers[-1].requires_grad_(True)
        llm.mamba.backbone.norm_f.requires_grad_(True)
        head = llm.mamba.lm_head
        head.weight = nn.Parameter(head.weight.data)
    else:
        llm = LitLLM(config=dict(cfg))

    # Start training
    trainer.fit(model=llm, datamodule=data, ckpt_path=cfg.resume_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainLLMConfig, train)
