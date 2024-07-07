from typing import Literal

import lightning.pytorch as pl
import torch


def configure_torch_backends(
    seed: int = 100,
    matmul_precision: Literal["medium", "high", "highest"] = "highest",
):
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)


def random_circular_crop(dna, L):
    start = torch.randint(len(dna), size=[1]).item()
    L = min(len(dna), L)
    crop = dna[start:(start + L)]
    overhang = L - len(crop)  # wrap around to start
    crop = crop + dna[:overhang]
    assert len(crop) == L
    return crop
