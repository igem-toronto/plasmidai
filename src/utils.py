from typing import Literal

import lightning.pytorch as pl
import torch


def configure_torch_backends(
    seed: int = 100,
    matmul_precision: Literal["medium", "high", "highest"] = "highest",
) -> None:
    """
    Configure PyTorch backend settings for reproducibility and performance.

    This function sets a random seed for PyTorch operations and configures
    the precision for float32 matrix multiplication.

    Args:
        seed (int): The random seed to set for PyTorch operations. Defaults to 100.
        matmul_precision (Literal["medium", "high", "highest"]): The precision level
            for float32 matrix multiplication. Defaults to "highest".
    """
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)


def random_circular_crop(dna: str, L: int) -> str:
    """
    Perform a random circular crop on a DNA sequence.

    This function takes a DNA sequence and a desired length, then returns
    a randomly selected contiguous subsequence of the specified length.
    If the desired length is greater than the input sequence length,
    it wraps around to the beginning of the sequence.

    Args:
        dna (str): The input DNA sequence to crop.
        L (int): The desired length of the cropped sequence.

    Returns:
        str: The randomly cropped DNA sequence of length L.

    Raises:
        AssertionError: If the length of the cropped sequence doesn't match L.
    """
    start: int = torch.randint(len(dna), size=[1]).item()
    L = min(len(dna), L)
    crop: str = dna[start : (start + L)]
    overhang: int = L - len(crop)  # wrap around to start
    crop = crop + dna[:overhang]
    assert (
        len(crop) == L
    ), f"Cropped sequence length {len(crop)} doesn't match desired length {L}"
    return crop
