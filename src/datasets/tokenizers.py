from typing import Optional

import torch

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


class BaseTokenizer:

    def __init__(self, vocab, sos_token, pad_token):
        self.vocab = vocab
        self.sos_token = sos_token
        self.pad_token = pad_token

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def sos_idx(self):
        return self.vocab[self.sos_token]

    @property
    def pad_idx(self):
        return self.vocab[self.pad_token]

    def tokenize(self, dna: str, L: Optional[int] = None):
        raise NotImplementedError()

    def decode(self, sequence: torch.Tensor, truncate: bool = True):
        raise NotImplementedError()


class NucleotideTokenizer(BaseTokenizer):

    def __init__(self):
        vocab = list("ACGT") + ["[SEP]", "[PAD]"]
        vocab = {token: i for i, token in enumerate(vocab)}
        super().__init__(vocab=vocab, sos_token="[SEP]", pad_token="[PAD]")

    def tokenize(self, dna: str, L: Optional[int] = None):
        tokens = []
        for x in dna:
            choices = LETTER_TO_BASES[x]
            i = torch.randint(len(choices), size=[1]).item()
            tokens.append(choices[i])
        tokens = [self.sos_token] + tokens + [self.sos_token]

        sequence = [self.vocab[tok] for tok in tokens]
        return torch.tensor(sequence, dtype=torch.long)

    def decode(self, sequence: torch.Tensor, truncate: bool = True):
        bocav = {i: tok for tok, i in self.vocab.items()}
        dna = []
        for idx in sequence:
            idx = idx.item()
            if truncate and (idx == self.vocab[self.sos_token]):
                break
            dna.append(bocav[idx])
        return " ".join(dna)


class PreTrainedTokenizer(BaseTokenizer):

    def __init__(self, tokenizer_path: str):
        from transformers import PreTrainedTokenizerFast

        # Load tokenizer from file
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        # Add special tokens to the tokenizer
        self.tokenizer.add_special_tokens({
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]"
        })

        super().__init__(
            vocab=self.tokenizer.vocab,
            sos_token="[CLS]",
            pad_token="[PAD]",
        )

        # TODO: temporary hack. In the future, only retain [SEP] and [PAD] tokens
        #   and featurize inputs as [SEP] ... [SEP]
        self.eos_token = "[SEP]"

    def tokenize(self, dna: str, L: Optional[int] = None):
        if L is None:
            truncation = False
            padding = False
        else:
            truncation = True
            padding = "max_length"

        # Encode
        return self.tokenizer(
            dna,
            max_length=L,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )["input_ids"][0]

    def decode(self, sequence: torch.Tensor, truncate: bool = True):
        dna = self.tokenizer.decode(sequence)
        if truncate:
            dna = dna.split(self.eos_token)[0]  # truncates at the first eos token
        return dna


def build_tokenizer(name) -> BaseTokenizer:
    if name == "nt":
        return NucleotideTokenizer()
    elif name.endswith(".json"):
        return PreTrainedTokenizer(name)
    else:
        raise ValueError()


if __name__ == "__main__":
    from src.paths import DATA_ROOT

    tok = build_tokenizer(str(DATA_ROOT / "tokenizer" / "dna_bpe_tokenizer_cutoff_rc.json"))
    print(tok.decode(tok.tokenize("ACGT"), truncate=False))
