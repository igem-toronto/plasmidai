from typing import Callable

import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None


def build_optimizer_and_scheduler(model, lr: Callable, betas, wd, **optim_kwargs):
    params = []
    params_no_wd = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        *attrs, name = name.split(".")

        # Get parent module
        parent = model
        for k in attrs:
            parent = getattr(parent, k)

        # Bucket parameters depending on whether they need weight decay
        if isinstance(parent, (nn.LayerNorm, RMSNorm)) or (name == "bias"):
            params_no_wd.append(p)
        elif getattr(p, "_no_weight_decay", False):  # some Mamba params.
            params_no_wd.append(p)
        else:
            params.append(p)

    optimizer = torch.optim.AdamW(
        params=[
            {"params": params, "weight_decay": wd},
            {"params": params_no_wd, "weight_decay": 0.0},
        ],
        lr=1.0, betas=betas,
        **optim_kwargs,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    return optimizer, scheduler
