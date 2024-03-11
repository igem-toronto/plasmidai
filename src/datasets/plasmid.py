import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from src.paths import DATA_ROOT
from src.utils import dna_to_tensor, random_circular_crop


class PlasmidDataset(Dataset):

    def __init__(self, records, Lmax):
        super().__init__()

        self.records = list(records)
        self.Lmax = Lmax

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # A=0 C=1 G=2 T=3
        sequence = random_circular_crop(str(record.seq), Lmax=self.Lmax)
        sequence = dna_to_tensor(sequence)
        if torch.rand(1) < 0.5:  # reverse-compliment
            sequence = 3 - sequence  # A->T C->G G->C T->A
        mask = torch.full_like(sequence, True, dtype=torch.bool)

        # Padding & mask
        pad = self.Lmax - sequence.shape[0]
        sequence = F.pad(sequence, pad=(0, pad))
        mask = F.pad(mask, pad=(0, pad), value=False)

        # Done!
        return sequence, mask


class PlasmidDataModule(pl.LightningDataModule):

    def __init__(self, Lmax=10000, batch_size=10, num_workers=0, finetune=False):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        root = DATA_ROOT / "plasmids"
        records = SeqIO.parse(root / f"plasmids.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(root / "splits.csv")
        if finetune:
            df = df[df["finetune"]]

        self.datasets = {}
        for split, group in df.groupby("split"):
            examples = [records[k] for k in group["id"]]
            self.datasets[split] = PlasmidDataset(examples, Lmax=Lmax)

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
            pin_memory=True,
        )
