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


class PlasmidTokenizer:

    def __init__(self):
        self.vocab = dict()
        self.eos = "eos"

        # Regular tokens
        self.index_to_base = "ACGT"
        for base in self.index_to_base:
            self._register_token(base)
        self._register_token(self.eos)

    def _register_token(self, name):
        self.vocab[name] = len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, dna, sos=True, eos=True):
        tokens = []
        if sos:
            tokens.append(self.eos)
        for x in dna:
            choices = LETTER_TO_BASES[x]
            i = torch.randint(len(choices), size=[1]).item()
            tokens.append(choices[i])
        if eos:
            tokens.append(self.eos)
        sequence = [self.vocab[tok] for tok in tokens]
        return torch.tensor(sequence, dtype=torch.long)

    def decode(self, sequence):
        assert sequence.ndim == 1
        assert torch.all((0 <= sequence) & (sequence < self.vocab_size))

        dna = []
        for idx in sequence:
            idx = idx.item()
            if idx == self.vocab[self.eos]:
                break
            dna.append(self.index_to_base[idx])
        return "".join(dna)


def random_circular_crop(dna, Lmax):
    start = torch.randint(len(dna), size=[1]).item()
    L = min(len(dna), Lmax)
    crop = dna[start:(start + L)]
    overhang = L - len(crop)  # wrap around to start
    crop = crop + dna[:overhang]
    assert len(crop) == L
    return crop
