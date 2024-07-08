from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

LETTER_TO_BASES = {
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

    def __init__(self, path: str):
        super().__init__(
            tokenizer_file=path,
            bos_token="[SEP]",
            eos_token="[SEP]",
            sep_token="[SEP]",
            pad_token="[PAD]",
        )

    def tokenize_dna(self, dna: str, max_length: Optional[int] = None):
        bases = []
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

    def decode_dna(self, token_ids):
        dna = self.decode(token_ids)
        dna = dna.split(self.eos_token)[1].strip()  # [SEP] A [SEP] -> A
        return dna
