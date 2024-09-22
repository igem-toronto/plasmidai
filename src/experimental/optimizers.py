from typing import Callable, List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    print("RMSNorm not available.")
    RMSNorm: Optional[type] = None


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: Callable[[int], float],
    betas: Tuple[float, float],
    wd: float,
    **optim_kwargs: Any
) -> Tuple[Optimizer, LambdaLR]:
    """
    Build an AdamW optimizer and a LambdaLR scheduler for a given model.

    This function separates parameters that require weight decay from those that don't,
    and creates an optimizer and scheduler accordingly.

    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        lr (Callable[[int], float]): A function that takes the current step and returns the learning rate.
        betas (Tuple[float, float]): Adam's beta parameters (beta1, beta2).
        wd (float): The weight decay to apply.
        **optim_kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        Tuple[Optimizer, LambdaLR]: A tuple containing the created optimizer and scheduler.
    """
    params: List[torch.Tensor] = []
    params_no_wd: List[torch.Tensor] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        *attrs, name = name.split(".")

        # Get parent module
        parent: nn.Module = model
        for k in attrs:
            parent = getattr(parent, k)

        # Bucket parameters depending on whether they need weight decay
        if isinstance(parent, (nn.LayerNorm, RMSNorm)) or (name == "bias"):
            params_no_wd.append(p)
        elif getattr(p, "_no_weight_decay", False):  # some Mamba params.
            params_no_wd.append(p)
        else:
            params.append(p)

    optimizer: Optimizer = torch.optim.AdamW(
        params=[
            {"params": params, "weight_decay": wd},
            {"params": params_no_wd, "weight_decay": 0.0},
        ],
        lr=1.0, betas=betas,
        **optim_kwargs,
    )

    scheduler: LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    return optimizer, scheduler