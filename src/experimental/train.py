from typing import Literal, Optional

import jsonargparse
import lightning.pytorch as pl

from src.datasets.plasmid import PlasmidDataModule
from src.experimental.callbacks import GradNormMonitor
from src.experimental.lit import LitLLM
from src.paths import LOG_DIR, random_checkpoint_dir
from src.utils import configure_torch_backends


class SimpleTrainer(pl.Trainer):

    def __init__(
        self,
        strategy: Literal["auto", "ddp"] = "auto",
        accelerator: Literal["cpu", "gpu"] = "cpu",
        devices: int = 1,
        precision: Literal["32", "16-mixed", "bf16-mixed", "bf16-true"] = "32",
        max_epochs: int = 50000,
        train_steps_per_epoch: int = 50,
        val_steps_per_epoch: int = 50,
        log_every_n_steps: int = 5,
        progress_bar: bool = False,
        wandb: bool = False,
        wandb_dir: str = str(LOG_DIR),
        wandb_project: str = "train_plasmid_llm",
        wandb_entity: Optional[str] = None,
        checkpoint: bool = False,
        checkpoint_dir: Optional[str] = random_checkpoint_dir(),
    ):
        callbacks = [pl.callbacks.ModelSummary(max_depth=2)]

        if checkpoint:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="epoch={epoch}-loss={val/loss_finetune:.3f}",
                    auto_insert_metric_name=False,
                    monitor="val/loss_finetune",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    verbose=True,
                )
            )

        if wandb:
            logger = pl.loggers.WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                log_model=False,
                save_dir=wandb_dir,
            )
            callbacks.extend([
                pl.callbacks.LearningRateMonitor(),
                GradNormMonitor(),
            ])
        else:
            logger = False

        super().__init__(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            callbacks=callbacks,
            enable_checkpointing=checkpoint,
            logger=logger,
            max_epochs=max_epochs,
            limit_train_batches=train_steps_per_epoch,
            limit_val_batches=val_steps_per_epoch,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=progress_bar,
            reload_dataloaders_every_n_epochs=1,
            use_distributed_sampler=True,
        )


def train():
    parser = jsonargparse.ArgumentParser()

    # Populate arguments
    parser.add_function_arguments(configure_torch_backends, "backend")
    parser.add_class_arguments(PlasmidDataModule, "data")
    parser.add_class_arguments(LitLLM, "lit")
    parser.add_class_arguments(SimpleTrainer, "trainer")
    parser.add_argument("--resume_path", type=Optional[str], default=None)
    parser.add_argument("--finetune_path", type=Optional[str], default=None)

    # Argument linking
    parser.link_arguments("data.Lmax", "lit.Lmax", apply_on="parse")

    # Parse
    cfg = parser.parse_args()

    # Instantiate
    init = parser.instantiate_classes(cfg)
    configure_torch_backends(**vars(cfg.backend))

    # Initialize and load model
    if cfg.finetune_path is not None:
        raise NotImplementedError()

    # Start training
    for logger in init.trainer.loggers:
        logger.log_hyperparams(cfg)
    init.trainer.fit(model=init.lit, datamodule=init.data, ckpt_path=cfg.resume_path)


if __name__ == "__main__":
    train()
