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

BASE_TO_INDEX = {b: i for i, b in enumerate("ACGT")}


def onehot_dna(sequence):
    onehot = []
    for x in sequence:
        choices = LETTER_TO_BASES[x]
        i = torch.randint(len(choices), size=[1]).item()
        base = choices[i]
        onehot.append(BASE_TO_INDEX[base])
    return torch.tensor(onehot, dtype=torch.long)


def random_roll(sequence):
    assert sequence.ndim == 1
    shift = torch.randint(sequence.shape[0], size=[1]).item()
    return torch.roll(sequence, shifts=shift, dims=0)
