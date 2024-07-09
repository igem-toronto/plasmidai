import lightning.pytorch as pl
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from src.datasets.utils import DNATokenizer
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
        return self.tokenizer.tokenize_dna(dna)


class RepliconDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokenizer_path: str = str(DATA_ROOT / "tokenizers" / "tokenizer_nt.json"),
        max_tokens: int = (2048 * 64),
        batch_size: int = 10,
        num_workers: int = 0,
    ):
        super().__init__()

        self.tokenizer_path = tokenizer_path
        self.tokenizer = DNATokenizer(self.tokenizer_path)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / f"replicons.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "replicons.splits.csv")

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

    # e.g., [SEP] A [SEP] B [SEP] C [SEP] D [SEP]
    def _collate(self, batch):
        x0 = batch.pop(0)
        batch = [x0] + [x[1:] for x in batch]
        batch = torch.cat(batch, dim=0)[:self.max_tokens]
        batch = batch.unsqueeze(0)  # (1 L C)
        return batch, torch.ones_like(batch, dtype=torch.bool)
