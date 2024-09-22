import torch
import torch.nn as nn

# remember that this is einstein operation, which is the special fancy way of reshaping.


from evo import Evo


class EvoMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.evo_full = Evo("evo-1-8k-base")

        self.evo_model, self.tokenizer = self.evo_full.model, self.evo_full.tokenizer
        # Freeze the layers of the evo model, also saves memory
        self.evo_freeze = self.config["evo_freeze"]
        if self.evo_freeze:
            for param in self.evo_model.parameters():
                param.requires_grad = False
            # just printing the parameters at the beginning, to know what to unfreeze

            # unfreeze some of the weights (the blocks 28 <)
            for i, (name, param) in enumerate(self.evo_model.named_parameters()):
                if self.config["full_debug"]:
                    print(f"index{i}")
                    print(f"Parameter name: {name}")
                    print(f"Parameter shape: {param.shape}")
                    print(f"Requires gradient: {param.requires_grad}")
                    print(param)
                    print("-" * 50)

        self.unfreeze_layers(self.config["unfreeze_index"])

    def unfreeze_layers(self, start_layer_idx):
        # Print all parameters in the model
        for idx, (name, param) in enumerate(self.evo_model.named_parameters()):
            if idx >= start_layer_idx:
                param.requires_grad = True
            if self.config["full_debug"]:
                print(f"Unfreezing {name}")

    def forward(self, x):
        if self.config["full_debug"]:
            print("Memory before evo (in MB)", torch.cuda.memory_allocated() / 1e6)
            print(
                "Here is the input format", x.shape
            )  # Here is the input format torch.Size([8, 512, 4096])

        x = self.evo_model(x)

        if self.config["full_debug"]:
            print("Final shape", x.shape)
            print("Memory after all (in MB)", torch.cuda.memory_allocated() / 1e6)

        return x


if __name__ == "__main__":
    # Example config dictionary
    config = {
        "evo_output_features": 512,
        "embed_channels": 256,
        "evo_freeze": True,
        "length_seq": 1024,
        "evo_input_embed_dim": 64,
        "evo_input_length": 16,
        "num_hidden_patch_embed": 2,
        "hidden_dim_patch_embed": 128,
        "dropout": 0.1,
        "full_debug": True,
    }

    # Instantiate the model
    model = EvoMamba(config)
