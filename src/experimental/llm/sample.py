from typing import Literal

import pydantic
import pydantic_cli
import pytorch_lightning as pl
import torch
import torch.backends.cuda
import torch.backends.cudnn
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader

from src.experimental.llm.lit import LitLLM
from src.utils import tensor_to_dna


class LLMSampleConfig(pydantic.BaseModel):

    seed: int = 100
    checkpoint_path: str

    accelerator: Literal["cpu", "gpu"] = "cpu"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"

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
        samples = self.model.generate(
            input_ids=batch,
            max_length=cfg.sample_max_length,
            top_k=cfg.sample_top_k,
            top_p=cfg.sample_top_p,
            min_p=cfg.sample_min_p,
            temperature=cfg.sample_temperature,
            vocab_size=(4 + 1),  # exclude sos
            eos_token_id=self.model.eos,
        )
        samples = samples[..., 1:]  # remove sos
        samples = [tensor_to_dna(x, eos=self.model.eos) for x in samples]
        return samples


def sample(config: LLMSampleConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load checkpoint
    lit = LitLLM.load_from_checkpoint(cfg.checkpoint_path, map_location="cpu")
    model = LLMSampler(cfg, lit)

    # Load dataset
    predict_loader = DataLoader(
        dataset=torch.full([cfg.num_samples, 1], lit.sos, dtype=torch.long),
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

    # Sample
    samples = trainer.predict(model, dataloaders=predict_loader)
    samples = sum(samples, [])  # flatten

    # Write to fasta
    records = []
    for i, plasmid in enumerate(samples):
        r = SeqRecord(seq=Seq(plasmid), id=f"sample_{i}", description="")
        records.append(r)
    SeqIO.write(records, cfg.samples_path, "fasta")

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(LLMSampleConfig, sample)
