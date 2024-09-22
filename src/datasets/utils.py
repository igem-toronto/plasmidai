from typing import Optional, Dict, List

import torch
from transformers import PreTrainedTokenizerFast

LETTER_TO_BASES: Dict[str, str] = {
    "A": "A",
    "B": "CGT",
    "C": "C",
    "D": "AGT",
    "G": "G",
    "H": "ACT",
    "K": "GT",
    "M": "AC",
    "N": "ACGT",
    "R": "AG",
    "S": "CG",
    "T": "T",
    "V": "ACG",
    "W": "AT",
    "Y": "CT",
}


class DNATokenizer(PreTrainedTokenizerFast):
    """
    A tokenizer for DNA sequences based on the PreTrainedTokenizerFast from Hugging Face.

    This tokenizer handles the conversion of DNA sequences to token IDs and vice versa,
    with support for ambiguous DNA bases and special tokens.

    Attributes:
        Inherits all attributes from PreTrainedTokenizerFast.
    """

    def __init__(self, path: str):
        """
        Initialize the DNATokenizer.

        Args:
            path (str): Path to the tokenizer file.
        """
        super().__init__(
            tokenizer_file=path,
            bos_token="[SEP]",
            eos_token="[SEP]",
            sep_token="[SEP]",
            pad_token="[PAD]",
        )

    def tokenize_dna(self, dna: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Tokenize a DNA sequence.

        This method handles ambiguous DNA bases by randomly choosing one of the possible bases.
        It then tokenizes the resulting sequence.

        Args:
            dna (str): The DNA sequence to tokenize.
            max_length (Optional[int]): The maximum length of the tokenized sequence.
                If None, no truncation or padding is applied.

        Returns:
            torch.Tensor: A tensor of token IDs representing the DNA sequence.
        """
        bases: List[str] = []
        for x in dna:
            choices = LETTER_TO_BASES[x.upper()]
            i = torch.randint(len(choices), size=[1]).item()
            bases.append(choices[i])
        dna = "".join(bases)

        if max_length is None:
            truncation = False
            padding = False
        else:
            truncation = True
            padding = "max_length"

        # Encode
        return self(
            dna,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )["input_ids"][0]

    def decode_dna(self, token_ids: torch.Tensor) -> str:
        """
        Decode a sequence of token IDs back into a DNA sequence.

        This method removes the special tokens and returns the pure DNA sequence.

        Args:
            token_ids (torch.Tensor): A tensor of token IDs to decode.

        Returns:
            str: The decoded DNA sequence.
        """
        dna = self.decode(token_ids)
        dna = dna.split(self.eos_token)[1].strip()  # [SEP] A [SEP] -> A
        return dna
