import pathlib
from typing import Literal

import einops
import pydantic
import pydantic_cli
import pytorch_lightning as pl
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader

from src.experimental.llm.lit import LitLLM
from src.utils import PlasmidTokenizer


class LLMSampleConfig(pydantic.BaseModel):
    seed: int = 100
    checkpoint_path: str

    accelerator: Literal["cpu", "gpu"] = "cpu"
    matmul_precision: Literal["medium", "high", "highest"] = "highest"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"

    # =============
    # Sample Fields
    # =============

    samples_path: str

    num_samples: int = 10000
    batch_size: int = 50

    sample_max_length: int = 10000
    sample_top_k: int = 4
    sample_top_p: float = 0.0
    sample_min_p: float = 0.0
    sample_temperature: float = 0.7

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


class LLMSampler(pl.LightningModule):

    def __init__(self, config: LLMSampleConfig, model):
        super().__init__()

        self.config = config
        self.model = model
        self.model.eval()
        self.tokenizer = PlasmidTokenizer()

        if config.precision.startswith("16"):
            self.model.half()
        elif config.precision.startswith("bf16"):
            self.model.bfloat16()
        else:
            self.model.float()

    def predict_step(self, batch):
        cfg = self.config
        samples = self.model.generate(
            input_ids=batch,
            max_length=cfg.sample_max_length,
            top_k=cfg.sample_top_k,
            top_p=cfg.sample_top_p,
            min_p=cfg.sample_min_p,
            temperature=cfg.sample_temperature,
            vocab_size=(4 + 1),
        )
        samples = samples[..., batch.shape[-1]:]  # remove prompt
        samples = [self.tokenizer.decode(x) for x in samples]
        return samples


def sample(config: LLMSampleConfig):
    cfg = config

    # Torch settings
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load checkpoint
    lit = LitLLM.load_from_checkpoint(cfg.checkpoint_path, map_location="cpu")
    model = LLMSampler(cfg, lit)

    # Load dataset
    prompts = model.tokenizer.tokenize("", eos=False)
    prompts = einops.repeat(prompts, "c -> n c", n=cfg.num_samples)
    predict_loader = DataLoader(
        dataset=prompts,
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
        log_every_n_steps=1,
    )

    # Sample
    samples = trainer.predict(model, dataloaders=predict_loader)
    samples = sum(samples, [])  # flatten

    # Write to fasta
    samples_path = pathlib.Path(cfg.samples_path)
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        SeqRecord(seq=Seq(plasmid), id=f"sample_{i}", description="")
        for i, plasmid in enumerate(samples)
    ]
    SeqIO.write(records, samples_path, "fasta")

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(LLMSampleConfig, sample)
