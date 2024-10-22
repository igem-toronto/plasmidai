from typing import List, Literal, Optional

import pytorch_lightning as pl
import pydantic_cli
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datasets import PlasmidDataModule
from src.experimental.callbacks import GradNormCallback
from src.experimental.d3pm.d3pm import D3PMConfig
from src.experimental.d3pm.lit import LitD3PM
from src.experimental.d3pm.lit import LitLLMConfig
from src.paths import LOG_DIR, random_checkpoint_dir


class TrainD3PMConfig(D3PMConfig, LitLLMConfig):

    seed: int = 100

    accelerator: str = "gpu"
    devices: int = 1
    strategy: Optional[str] = "auto"

    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"

    # =================
    # Datamodule Fields
    # =================

    plasmid_length: int = 10000

    batch_size: int = 32
    num_workers: int = 6
    finetune: bool = False

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = 1500
    train_steps_per_epoch: int = 50
    val_steps_per_epoch: int = 50

    resume_path: Optional[str] = None

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_project: str = "train_plasmid_d3pm"
    wandb_entity: Optional[str] = None
    wandb_dir: Optional[str] = None

    checkpoint: bool = False
    checkpoint_dir: Optional[str] = None

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainD3PMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    data = PlasmidDataModule(
        Lmax=cfg.plasmid_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        finetune=cfg.finetune,
    )

    # Initialize trainer
    callbacks = [
        ModelSummary(max_depth=2),
        GradNormCallback(),
    ]

    if cfg.wandb:
        logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=LOG_DIR,
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
                monitor="val/loss",
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
    d3pm = LitD3PM(config=dict(cfg))

    # Start training
    trainer.fit(model=d3pm, datamodule=data, ckpt_path=cfg.resume_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainD3PMConfig, train)
