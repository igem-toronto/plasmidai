"""
Implementation from the mamba blocks to all the possible tokens for classification
"""
import torch.nn as nn
import torch


class TokenRegressor(nn.Module):
    """Final regressor from mamba blocks to tokens"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = self.config["mamba_outputs"]
        self.output_dim = self.config["num_tokens"]
        self.hidden_dim = self.config["hidden_dim"]
        self.num_hidden_layers = self.config["num_hidden_layers"]
        self.dropout = self.config["dropout"]
        if self.dropout > 0.0:
            self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        self.regressor = self.create_model()

    def create_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))

        for _ in range(self.num_hidden_layers):
            if self.dropout > 0:
                layers.append(self.dropout_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.extend([nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.regressor(x)
        return x
