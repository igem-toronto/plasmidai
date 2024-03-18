import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from src.paths import DATA_ROOT
from src.utils import PlasmidTokenizer, random_circular_crop


class PlasmidDataset(Dataset):

    def __init__(self, records, Lmax):
        super().__init__()

        self.records = list(records)
        self.Lmax = Lmax  # max number of nt in input sequence

        self.tokenizer = PlasmidTokenizer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Crop & augment
        dna = random_circular_crop(record.seq, Lmax=self.Lmax)  # Bio.Seq object
        dna = dna.reverse_complement()
        dna = str(dna)

        # Tokenize
        # [eos, nt(1), nt(2), ..., nt(Lmax-1), (eos or nt(Lmax))]
        # depending on whether plasmid has length < Lmax
        dna = self.tokenizer.tokenize(dna, eos=(len(dna) < self.Lmax))
        mask = torch.full(dna.shape, True)

        # Padding to Lmax + 1
        pad = (self.Lmax + 1) - dna.shape[0]
        dna = F.pad(dna, pad=(0, pad))
        mask = F.pad(mask, pad=(0, pad), value=False)

        # Done!
        return dna, mask, record.finetune


class PlasmidDataModule(pl.LightningDataModule):

    def __init__(self, Lmax=10000, batch_size=10, num_workers=0, finetune=False):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / f"plasmids.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "splits.csv")
        for _, row in df.iterrows():
            records[row["id"]].finetune = row["finetune"]
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
