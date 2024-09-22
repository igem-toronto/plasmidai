import torch
import torch.optim as optim

import os
from einops import rearrange


def configure_optim_scheduler(config, model):
    """Returns the optimizer and scheduler functions

    Args:
        config (dict):configuration file information
        model (nn.Module): model used

    Returns:
        Tuple [
            optimizer (torch.optim)
            scheduler (optim.lr_scheduler)
        ]
    """
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
        )

    elif config["optimizer"] == "adamW":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

    else:
        print("No optimizers selected!")
        raise NotImplementedError

    # learning rate scheduler
    if config["scheduler_fct"] == "RRLP":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5
        )  # half the learning rate each time

    if config["scheduler_fct"] == "cosine":
        for param_group in optimizer.param_groups:
            param_group.setdefault("initial_lr", config["learning_rate"])
        T_0 = config["T_0"]
        T_mult = config["T_mult"]  # doubling the spacing bewteen each reset
        eta_min = config["eta_min"]  # the minimum learning rate
        last_epoch = config["epoch_number"]
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    return optimizer, scheduler


def run_step(model, epoch, first_epoch, config, data, loss_fn, batch_idx, split):
    """Train one epoch
    Args:
        model (nn.Module)
        epoch (int): epoch number
        first_epoch (bool): whether the current epoch is the first
        config (dict): configuration dictionary
        data (DataLoader): dataset
        loss_fn (nn.Module)
        batch_idx (int): index of the current batch within the epoch

    Returns:
        Tuple[float, float]: [train loss per batch, total train loss]
    """
    print(f"[INFO] Epoch {epoch} | {split} set: ")

    input_batch, mask = data  # (B L+1) (B L+1) (B)

    mask = mask[..., 1:]  # (B L)
    num_tokens = mask.int().sum()

    if first_epoch and batch_idx == 0:
        print("[INFO] INPUT shape: ", input_batch.shape)
        print("[INFO] MASK shape:", mask.shape)
        print("[INFO] Number of tokens: ", num_tokens)

    logits, _ = model(input_batch[..., :-1])  # (B L C)
    logits = rearrange(logits, "b l c -> b c l")

    outputs, labels = logits, input_batch[..., 1:]

    if first_epoch and batch_idx == 0:
        print("[INFO] OUTPUT shape: ", outputs.shape)
        print("[INFO] LABELS shape: ", labels.shape)
        print(f"[INFO] Sample OUTPUT: {outputs[0][0]}")
        print(f"[INFO] Sample LABEL: {labels[0]}")

    outputs = outputs  # was not one-hot encoded, but is a probabilistic distribution

    # broadcast the loss
    losses = loss_fn(outputs, labels)

    # compute the loss where the mask is True
    losses = torch.where(mask, losses, 0)

    loss = losses.sum() / num_tokens.float()

    if first_epoch and batch_idx == 0:
        print(f"[INFO] Memory usage per step: {torch.cuda.memory_reserved()/1e6} (MB)")

    print(f"[INFO] Training loss: {loss}")

    torch.cuda.empty_cache()
    return loss


def load_weights(model, checkpoint_path):
    """Returns the model with weights loaded

    Args:
        model (nn.Module): Initialized model
        checkpoint_path (str): path of the checkpoint
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )

        print(f"[INFO] checkpoint type: {type(checkpoint)}")
        print(f"[INFO] checkpoint keys: {checkpoint.keys()}")

        # checkpoint state_dict
        state_dict = checkpoint
        # print('[INFO] checkpoint model_state dict: ', state_dict)
        print(f"[INFO] type(model) {type(model)}")

        # model state_dict
        model_state_dict = model.state_dict()

        # comparing the dimensions, and only keeping the state_dict values that have same dimensions
        merged_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict
            and k in state_dict
            and model_state_dict[k].shape == state_dict[k].shape
        }

        if merged_state_dict != state_dict:
            print(
                "[WARNING] The weights of you model had different shapes from the weights that you loaded"
            )
            print(
                "[WARNING] Here are the differing parameters: ",
                model_state_dict.keys() ^ state_dict.keys(),
            )

        model.load_state_dict(merged_state_dict, strict=False)
        print("[INFO] The weights for your model have been successfully loaded")

    else:
        print("[WARN] Checkpoint path is empty. Recheck the checkpoint")

    return model
