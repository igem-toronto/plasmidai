from typing import Optional

import torch
from transformers import BatchEncoding, PreTrainedTokenizerFast

from src.paths import DATA_ROOT


class PlasmidTokenizer:
    """A tokenizer class for handling DNA sequences using Byte Pair Encoding (BPE).
    """

    def __init__(self, tokenizer_path: str):
        # Load tokenizer from file
        self.tokenizer_path = str(tokenizer_path)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)

        # Define special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"

        # Create a dictionary of the special tokens
        self.special_tokens = {
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token
        }

        # Add special tokens to the tokenizer
        self.tokenizer.add_special_tokens(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def vocab(self) -> dict:
        return self.tokenizer.vocab

    def tokenize(
        self,
        dna: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: str = "max_length",
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        return self.tokenizer(
            dna,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=return_tensors,
        )

    def decode(self, sequence: torch.Tensor, truncate: bool = True) -> str:
        output = self.tokenizer.decode(sequence)
        if truncate:
            output = output.split(self.sep_token)[0]  # truncates at the first [SEP] token
        return output


TOKENIZER = PlasmidTokenizer(DATA_ROOT / "tokenizer" / "dna_bpe_tokenizer_cutoff_rc.json")


def random_circular_crop(dna, Lmax):
    start = torch.randint(len(dna), size=[1]).item()
    L = min(len(dna), Lmax)
    crop = dna[start:(start + L)]
    overhang = L - len(crop)  # wrap around to start
    crop = crop + dna[:overhang]
    assert len(crop) == L
    return crop
