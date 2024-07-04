from typing import Literal, Optional

import jsonargparse
import torch
import tqdm
import wandb

from src.experimental.lit import LitLLM
from src.paths import LOG_DIR
from src.utils import configure_torch_backends


@torch.no_grad()
def sample_loop(
    checkpoint_path: str,
    precision: Literal["float", "half", "bfloat16"] = "float",
    num_samples: int = 10000,
    batch_size: int = 50,
    top_k: int = -1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    wandb_dir: str = str(LOG_DIR),
    wandb_project: str = "sample_plasmid_llm",
    wandb_entity: Optional[str] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, precision)

    sample_kwargs = dict(
        checkpoint_path=checkpoint_path,
        num_samples_per_epoch=batch_size,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    lit: LitLLM = LitLLM.load_from_checkpoint(**sample_kwargs, map_location="cpu")
    lit.to(device=device, dtype=dtype)
    lit.eval()

    samples = []
    with torch.autocast(device_type=device, dtype=dtype):
        for _ in tqdm.trange(num_samples // batch_size, desc="Sampling"):
            samples += [[x.replace(" ", "")] for x in lit._sample()]
    table = wandb.Table(columns=["sequence"], data=samples)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        dir=wandb_dir,
        config=dict(**sample_kwargs, precision=precision)
    )
    wandb.log({"samples": table})
    wandb.finish()


def sample():
    parser = jsonargparse.ArgumentParser()

    # Populate arguments
    parser.add_function_arguments(configure_torch_backends, "backend")
    parser.add_function_arguments(sample_loop, "sample")

    # Parse
    cfg = parser.parse_args()

    # Call
    configure_torch_backends(**vars(cfg.backend))
    sample_loop(**vars(cfg.sample))


if __name__ == "__main__":
    sample()
