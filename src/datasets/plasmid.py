import lightning.pytorch as pl
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

from src.datasets.utils import DNATokenizer
from src.paths import DATA_ROOT
from src.utils import random_circular_crop


class PlasmidDataset(Dataset):
    """
    A dataset for plasmid DNA sequences.

    This dataset handles loading, preprocessing, and tokenization of plasmid DNA sequences.
    It supports random circular cropping and reverse complement augmentation.

    Attributes:
        records (List[SeqRecord]): List of SeqRecord objects containing plasmid sequences.
        tokenizer (DNATokenizer): Tokenizer for converting DNA sequences to token IDs.
        Lmax (int): Maximum number of tokens in the input sequence.
        Lcrop (int): Maximum length of DNA that can be produced by Lmax tokens.
    """

    def __init__(self, records: List[SeqRecord], tokenizer: DNATokenizer, Lmax: int):
        super().__init__()

        self.records = list(records)
        self.tokenizer = tokenizer
        self.Lmax = Lmax  # max number of nt in input sequence

        # Maximum length of DNA that can be produced by Lmax tokens
        self.Lcrop = Lmax * max(len(x) for x in self.tokenizer.vocab)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]

        # Crop & augment
        dna = random_circular_crop(record.seq, L=self.Lcrop)  # Bio.Seq object
        if torch.rand(1) < 0.5:
            dna = dna.reverse_complement()
        dna = str(dna)

        # Tokenize
        sequence = self.tokenizer.tokenize_dna(dna, max_length=self.Lmax)
        mask = sequence != self.tokenizer.pad_token_id
        return sequence, mask


class PlasmidDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for plasmid DNA sequences.

    This DataModule handles the loading, splitting, and preparation of plasmid DNA data
    for training, validation, and testing.

    Attributes:
        tokenizer_path (str): Path to the tokenizer file.
        tokenizer (DNATokenizer): Tokenizer for converting DNA sequences to token IDs.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        datasets (Dict[str, PlasmidDataset]): Dictionary of datasets for each split.
    """

    def __init__(
        self,
        tokenizer_path: str = str(
            DATA_ROOT / "tokenizers" / "dna_bpe_tokenizer_cutoff_rc.json"
        ),
        Lmax: int = 2048,
        batch_size: int = 10,
        num_workers: int = 0,
        finetune: bool = False,
    ):
        super().__init__()

        self.tokenizer_path = tokenizer_path
        self.tokenizer = DNATokenizer(self.tokenizer_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        records = SeqIO.parse(DATA_ROOT / "plasmids.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "plasmids.splits.csv")
        if finetune:
            df = df[df["finetune"]]

        self.datasets: Dict[str, PlasmidDataset] = {}
        for split, group in df.groupby("split"):
            examples = [records[k] for k in group["id"]]
            self.datasets[split] = PlasmidDataset(examples, self.tokenizer, Lmax=Lmax)

    def train_dataloader(self) -> DataLoader:
        return self._loader(split="train")

    def val_dataloader(self) -> DataLoader:
        return self._loader(split="val")

    def test_dataloader(self) -> DataLoader:
        return self._loader(split="test")

    def _loader(
        self, split: str, shuffle: bool = True, drop_last: bool = True
    ) -> DataLoader:
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
        )
