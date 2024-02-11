import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset, random_split

from src.datasets.utils import onehot_dna, random_roll
from src.paths import DATA_ROOT


class PlasmidDataset(Dataset):

    def __init__(self, records):
        super().__init__()

        self.records = list(records)
        self.Lmax = max(len(r.seq) for r in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # A=0 C=1 G=2 T=3
        sequence = onehot_dna(str(record.seq))
        sequence = random_roll(sequence)
        mask = torch.full_like(sequence, True, dtype=torch.bool)

        # Padding & mask
        pad = self.Lmax - sequence.shape[0]
        sequence = F.pad(sequence, pad=(0, pad))
        mask = F.pad(mask, pad=(0, pad), value=False)

        # Done!
        return sequence, mask


class PlasmidDataModule(pl.LightningDataModule):

    def __init__(
        self,
        seed=0,
        batch_size=10,
        num_workers=0,
        split_ratio=(0.8, 0.1, 0.1),
        split_by="random",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / f"plasmids.fasta", "fasta")
        self.base = PlasmidDataset(records)

        if split_by == "random":
            g = torch.Generator().manual_seed(seed)
            splits = random_split(self.base, split_ratio, generator=g)
            self.train_set, self.valid_set, self.test_set = splits
        else:
            raise NotImplementedError()

    def train_dataloader(self):
        return self._loader(self.train_set)

    def val_dataloader(self):
        return self._loader(self.valid_set)

    def test_dataloader(self):
        return self._loader(self.test_set)

    def _loader(self, dataset, shuffle=True, drop_last=True):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
        )
