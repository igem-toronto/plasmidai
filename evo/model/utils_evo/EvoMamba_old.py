import torch
import torch.nn as nn

# remember that this is einstein operation, which is the special fancy way of reshaping.
from einops import rearrange


from model import VideoMamba as vm

from evo import Evo
from model.FinalRegressor import TokenRegressor


class EvoMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.evo_full = Evo("evo-1-8k-base")
        self.evo_model, self.tokenizer = self.evo_full.model, self.evo_full.tokenizer

        # Freeze the layers of the evo model, also saves memory
        self.evo_freeze = self.config["evo_freeze"]
        if self.evo_freeze:
            for param in self.evo_model.parameters():
                param.requires_grad = False

        self.patch_embedding = PatchEmbed(self.config)

        self.mamba = vm.videomamba_tiny(
            output_evo_num_channels=self.config["evo_output_features"],
            embed_dim=self.config["embed_channels"],
        )

        self.regressor = TokenRegressor(self.config)

    def forward(self, x):
        x = self.patch_embedding(x)
        # I don't think I need a patch_embedding before evo?? because it tells me that it is expecting long tensor...

        if self.config["full_debug"]:
            print("Memory before evo (in MB)", torch.cuda.memory_allocated() / 1e6)
            print(
                "Here is the input format", x.shape
            )  # Here is the input format torch.Size([8, 512, 4096])

        # somehow expects long:
        x = x.long()
        print(x)
        x = self.evo_model(x)

        if self.config["full_debug"]:
            print("Memory before mamba (in MB)", torch.cuda.memory_allocated() / 1e6)
            print("Here is the input format", x.shape)

        x = self.mamba(x)

        if self.config["full_debug"]:
            print("Transition shape", x.shape)
            print("Memory after mamba (in MB)", torch.cuda.memory_allocated() / 1e6)

        x = self.regressor(x)

        if self.config["full_debug"]:
            print("Final shape", x.shape)
            print("Memory after all (in MB)", torch.cuda.memory_allocated() / 1e6)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_sequence_length = self.config["length_seq"]  # where length_seq
        self.evo_input_embed_dim = self.config["evo_input_embed_dim"]
        self.evo_input_length = self.config["evo_input_length"]
        self.num_hidden_layers = self.config["num_hidden_patch_embed"]

        self.hidden_dim = self.config["hidden_dim_patch_embed"]
        self.output_dim = self.evo_input_embed_dim * self.evo_input_length

        self.proj = self.regressor()

        # adding opsitional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.evo_input_embed_dim, self.evo_input_length)
        )

        # usually no dropout in the projection layers
        self.dropout = self.config["dropout"]
        if self.dropout > 0.0:
            self.dropout_layer = torch.nn.Dropout(p=self.dropout)

    def regressor(self):
        layers = []
        layers.append(nn.Linear(self.input_sequence_length, self.hidden_dim))

        for _ in range(self.num_hidden_layers):
            # if self.dropout > 0:
            # layers.append(self.dropout_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.extend([nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(
            x, "b (c d) -> b c d", c=self.evo_input_embed_dim, d=self.evo_input_length
        )

        # adding positional embedding
        x = x + self.pos_embed
        return x
