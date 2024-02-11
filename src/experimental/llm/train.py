from typing import List, Literal, Optional

import lightning.pytorch as pl
import pydantic_cli
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

from src.datasets import PlasmidDataModule
from src.experimental.callbacks import GradNormCallback
from src.experimental.llm.lit import LitLLM, LitLLMConfig
from src.paths import LOG_DIR, random_checkpoint_dir


class TrainLLMConfig(LitLLMConfig):
    #
    seed: int = 100

    accelerator: str = "cpu"
    devices: int = 1
    strategy: Optional[str] = "auto"

    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"

    # =================
    # Datamodule Fields
    # =================

    batch_size: int = 32
    num_workers: int = 8
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    split_by: str = "random"

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
    wandb_project: str = "train_plasmid_llm"
    wandb_entity: Optional[str] = None

    checkpoint: bool = False
    checkpoint_dir: Optional[str] = None

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainLLMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    data = PlasmidDataModule(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_ratio=cfg.split_ratio,
        split_by=cfg.split_by,
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
    llm = LitLLM(config=dict(cfg))

    # Start training
    trainer.fit(model=llm, datamodule=data, ckpt_path=cfg.resume_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainLLMConfig, train)
