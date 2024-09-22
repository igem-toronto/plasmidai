"""After facing many issues with DDP, switched to lightning"""
import torch
import lightning as L

# from evo import Evo
from model.evo_utils.load_evo import Evo  # run from source
from utils.learning import configure_optim_scheduler, run_step, load_weights
from loss.EvoLoss import EvoLoss


class LitEvo(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # using the import configuration
        self.evo_full = Evo(
            model_name="evo-1-8k-base",
            config_path=config["evo_config_path"],
            model_path=config["evo_model_path"],
        )
        self.evo_model, self.tokenizer = self.evo_full.model, self.evo_full.tokenizer

        self.evo_model = load_weights(self.evo_model, self.config["evo_weights_path"])
        self.evo_model = self.evo_model.to(torch.float16)

        if self.config["evo_freeze"]:
            self.freeze_layers()
            self.unfreeze_layers(self.config["unfreeze_index"])

        self.loss_fn = EvoLoss(self.config)

    def forward(self, x):
        """Forward function

        Args:
            x: batch of data from DataLoader

        """
        x = self.evo_model(x)
        return x

    def unfreeze_layers(self, start_layer_idx):
        """Unfreeze certain layers of the model

        Args:
            start_layer_idx (int): index of the block which wshould not be frozen
        """
        for idx, (name, param) in enumerate(self.evo_model.named_parameters()):
            if idx >= start_layer_idx:
                param.requires_grad = True
            if self.config["full_debug"]:
                print(f"Unfreezing {name}")

    def freeze_layers(self):
        """Freeze all the layers in the evo model"""
        for param in self.evo_model.parameters():
            param.requires_grad = False

        for i, (name, param) in enumerate(self.evo_model.named_parameters()):
            if self.config["full_debug"]:
                print(f"index{i}")
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Requires gradient: {param.requires_grad}")
                print(param)
                print("-" * 50)

    def training_step(self, data, batch_idx):
        """Training step

        Args:
            data (Any[numpy.array, DataLoader]): Any batch of data from the common dataloaders
            batch_idx (int): integrer that represents the index of the the batch
        """
        kwargs = {
            "model": self,
            "epoch": self.current_epoch,
            "first_epoch": self.current_epoch == 0,
            "config": self.config,
            "data": data,
            "loss_fn": self.loss_fn,
            "batch_idx": batch_idx,
        }
        loss = run_step(**kwargs, split="train")

        self.log(
            "[INFO] Train Loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, data, batch_idx):
        """Validation step

        Args:
            data (Any[numpy.array, DataLoader]): Any batch of data from the common dataloaders
            batch_idx (int): integrer that represents the index of the the batch
        """
        kwargs = {
            "model": self,
            "epoch": self.current_epoch,
            "first_epoch": self.current_epoch == 0,
            "config": self.config,
            "data": data,
            "loss_fn": self.loss_fn,
            "batch_idx": batch_idx,
        }
        loss = run_step(**kwargs, split="validation")

        self.log(
            "[INFO] Validation Loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure the optimizers the of the following training loop"""
        optimizer, scheduler = configure_optim_scheduler(self.config, self)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
