import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EvoLoss(nn.Module):
    """I still need to do a bit more reading, but I am guessing they have different losses"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perplexity_loss = self.perplexity
        # self.crossentropy_loss = self.cross_entropy_loss
        self.crossentropy_loss = (
            F.cross_entropy
        )  # (logits, dnas[..., 1:], reduction="none")  # (... L)

    def cross_entropy_loss(self, expected_output, predicted_output):
        """
        Compute the cross-entropy loss between expected_output and predicted_output.

        Args:
            expected_output (numpy.ndarray): The expected output.
            predicted_output (numpy.ndarray): The predicted output.

        Returns:
            float: The cross-entropy loss.
        """
        # Ensure inputs are numpy arrays
        expected_output = np.array(expected_output)
        predicted_output = np.array(predicted_output)

        # Compute cross-entropy loss
        loss = -np.sum(expected_output * np.log(predicted_output))

        return loss

    def perplexity(self, expected_output, predicted_output):
        """
        Compute the perplexity between expected_output and predicted_output.

        Args:
            expected_output (numpy.ndarray): The expected output.
            predicted_output (numpy.ndarray): The predicted output.

        Returns:
            float: The perplexity.
        """
        # Compute cross-entropy loss
        loss = self.cross_entropy_loss(expected_output, predicted_output)

        # Compute perplexity
        perplexity = np.exp(loss)

        return perplexity

    def forward(self, expected_output, predicted_output):
        computed_loss = 0.0
        if "perplexity" in self.config["loss_type"]:
            computed_loss += (
                self.perplexity_loss(expected_output, predicted_output)
                * self.config["loss_type"]["perplexity"]
            )

        if "crossentropy" in self.config["loss_type"]:
            computed_loss += (
                self.crossentropy_loss(
                    expected_output, predicted_output, reduction="none"
                )
                * self.config["loss_type"]["crossentropy"]
            )
        return computed_loss
