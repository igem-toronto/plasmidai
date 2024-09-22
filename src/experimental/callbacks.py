import lightning.pytorch as pl
from lightning.pytorch import LightningModule


class GradNormMonitor(pl.Callback):
    """
    A PyTorch Lightning callback that monitors the L2 norm of gradients during training.
    """

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: LightningModule
    ) -> None:
        """
        Compute and log the L2 norm of gradients after the backward pass.

        This method is called automatically by PyTorch Lightning after each backward pass
        during training. It computes the L2 norm of the gradients for all parameters
        in the model and logs it using the module's logger.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning module being trained.
        """
        grad_2norm: float = pl.utilities.grad_norm(pl_module, norm_type=2.0)[
            "grad_2.0_norm_total"
        ]
        pl_module.log("grad_2norm", grad_2norm)
