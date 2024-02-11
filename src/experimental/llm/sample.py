from typing import Literal

import lightning.pytorch as pl
import pydantic
import pydantic_cli
import torch
import torch.backends.cuda
import torch.backends.cudnn
from torch.utils.data import DataLoader

from src.experimental.llm.lit import LitLLM


class LLMSampleConfig(pydantic.BaseModel):

    seed: int = 100
    checkpoint_path: str

    accelerator: Literal["cpu", "gpu"] = "cpu"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "bf16-mixed"

    # =============
    # Sample Fields
    # =============

    samples_path: str

    num_samples: int = 1000
    batch_size: int = 50

    sample_max_length: int = 11000
    sample_top_k: int = 0
    sample_top_p: float = 0.0
    sample_min_p: float = 0.0
    sample_temperature: float = 1.0

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


class LLMSampler(pl.LightningModule):

    def __init__(self, config: LLMSampleConfig, model):
        super().__init__()

        self.config = config
        self.model = model
        self.model.eval()

    def predict_step(self, batch):
        cfg = self.config

        sos = torch.full([batch.shape[0], 1], self.model.sos, device=self.device)
        samples = self.model.generate(
            input_ids=sos,
            max_length=cfg.sample_max_length,
            top_k=cfg.sample_top_k,
            top_p=cfg.sample_top_p,
            min_p=cfg.sample_min_p,
            temperature=cfg.sample_temperature,
            vocab_size=(4 + 1),  # exclude sos
            eos_token_id=self.model.eos,
        )

        print(samples, samples.shape, samples.dtype)
        exit()


def sample(config: LLMSampleConfig):
    cfg = config

    # Torch settings
    if cfg.accelerator == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load checkpoint
    model = LitLLM.load_from_checkpoint(cfg.checkpoint_path, map_location="cpu")
    model = LLMSampler(cfg, model)

    # Load dataset
    predict_loader = DataLoader(
        dataset=list(range(cfg.num_samples)),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        strategy="auto",
        devices=1,
        precision=cfg.precision,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        use_distributed_sampler=True,
    )

    # Start predicting
    samples = trainer.predict(model, dataloaders=predict_loader)
    print(samples)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(LLMSampleConfig, sample)
