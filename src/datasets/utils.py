import torch

BASE_TO_ONEHOT = torch.full([90], -100, dtype=torch.long)
BASE_TO_ONEHOT[[ord(base) for base in "ACGT"]] = torch.arange(4).long()


def onehot_dna(sequence):
    sequence = sequence.encode("ascii")
    return BASE_TO_ONEHOT[memoryview(sequence)]


def random_roll(sequence):
    assert sequence.ndim == 1
    shift = torch.randint(sequence.shape[0], size=[1]).item()
    return torch.roll(sequence, shifts=shift, dims=0)
