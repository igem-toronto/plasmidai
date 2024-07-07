import lightning.pytorch as pl
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from src.datasets.tokenizers import build_tokenizer
from src.paths import DATA_ROOT


class RepliconDataset(Dataset):

    def __init__(self, records, tokenizer):
        super().__init__()

        self.records = list(records)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Crop & augment
        dna = record.seq
        if torch.rand(1) < 0.5:
            dna = dna.reverse_complement()
        dna = str(dna)

        # Tokenize
        return self.tokenizer.tokenize(dna)[0]


class RepliconDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokenizer: str = "nt",
        max_tokens: int = 131072,
        batch_size: int = 10,
        num_workers: int = 0,
    ):
        super().__init__()

        self.tokenizer = build_tokenizer(tokenizer)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / f"plasmids.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "splits.csv")

        self.datasets = {}
        for split, group in df.groupby("split"):
            examples = [records[k] for k in group["id"]]
            self.datasets[split] = RepliconDataset(examples, self.tokenizer)

    def train_dataloader(self):
        return self._loader(split="train")

    def val_dataloader(self):
        return self._loader(split="val")

    def test_dataloader(self):
        return self._loader(split="test")

    def _loader(self, split, shuffle=True, drop_last=True):
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def _collate(self, batch):
        pass
