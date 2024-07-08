import lightning.pytorch as pl
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from src.datasets.utils import DNATokenizer
from src.paths import DATA_ROOT
from src.utils import random_circular_crop


class PlasmidDataset(Dataset):

    def __init__(self, records, tokenizer, Lmax):
        super().__init__()

        self.records = list(records)
        self.tokenizer = tokenizer
        self.Lmax = Lmax  # max number of nt in input sequence

        # Maximum length of DNA that can be produced by Lmax tokens
        self.Lcrop = Lmax * max(len(x) for x in self.tokenizer.vocab)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Crop & augment
        dna = random_circular_crop(record.seq, L=self.Lcrop)  # Bio.Seq object
        if torch.rand(1) < 0.5:
            dna = dna.reverse_complement()
        dna = str(dna)

        # Tokenize
        sequence = self.tokenizer.tokenize_dna(dna, max_length=self.Lmax)
        mask = (sequence != self.tokenizer.pad_token_id)
        return sequence, mask


class PlasmidDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokenizer: str = str(DATA_ROOT / "tokenizers" / "dna_bpe_tokenizer_cutoff_rc.json"),
        Lmax: int = 2048,
        batch_size: int = 10,
        num_workers: int = 0,
        finetune: bool = False,
    ):
        super().__init__()

        self.tokenizer = DNATokenizer(tokenizer)
        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / f"plasmids.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "plasmids.splits.csv")
        if finetune:
            df = df[df["finetune"]]

        self.datasets = {}
        for split, group in df.groupby("split"):
            examples = [records[k] for k in group["id"]]
            self.datasets[split] = PlasmidDataset(examples, self.tokenizer, Lmax=Lmax)

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
