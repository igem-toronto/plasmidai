import lightning.pytorch as pl
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

from src.datasets.utils import DNATokenizer
from src.paths import DATA_ROOT


class RepliconDataset(Dataset):
    """
    A dataset for replicon DNA sequences.

    This dataset handles loading, preprocessing, and tokenization of replicon DNA sequences.
    It supports reverse complement augmentation.

    Attributes:
        records (List[SeqRecord]): List of SeqRecord objects containing replicon sequences.
        tokenizer (DNATokenizer): Tokenizer for converting DNA sequences to token IDs.
    """

    def __init__(self, records: List[SeqRecord], tokenizer: DNATokenizer):
        super().__init__()

        self.records = list(records)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        record = self.records[idx]

        # Crop & augment
        dna = record.seq
        if torch.rand(1) < 0.5:
            dna = dna.reverse_complement()
        dna = str(dna)

        # Tokenize
        return self.tokenizer.tokenize_dna(dna)


class RepliconDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for replicon DNA sequences.

    This DataModule handles the loading, splitting, and preparation of replicon DNA data
    for training, validation, and testing.

    Attributes:
        tokenizer_path (str): Path to the tokenizer file.
        tokenizer (DNATokenizer): Tokenizer for converting DNA sequences to token IDs.
        max_tokens (int): Maximum number of tokens in a batch.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        datasets (Dict[str, RepliconDataset]): Dictionary of datasets for each split.
    """

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

        records = SeqIO.parse(DATA_ROOT / "replicons.fasta", "fasta")
        records = {r.id: r for r in records}

        df = pd.read_csv(DATA_ROOT / "replicons.splits.csv")

        self.datasets: Dict[str, RepliconDataset] = {}
        for split, group in df.groupby("split"):
            examples = [records[k] for k in group["id"]]
            self.datasets[split] = RepliconDataset(examples, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(split="val", shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(split="test", shuffle=False, drop_last=False)

    def _loader(
        self, split: str, shuffle: bool = True, drop_last: bool = True
    ) -> DataLoader:
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def _collate(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function for batching replicon sequences.

        This function combines multiple sequences into a single batch, truncating if necessary.
        It also creates a mask tensor for the batch.

        Args:
            batch (List[torch.Tensor]): List of tokenized sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batched sequences and corresponding mask.
        """
        x0 = batch.pop(0)
        batch = [x0] + [x[1:] for x in batch]
        batch = torch.cat(batch, dim=0)[: self.max_tokens]
        batch = batch.unsqueeze(0)  # (1 L C)
        return batch, torch.ones_like(batch, dtype=torch.bool)
