import lightning.pytorch as pl
import lightning.pytorch.utilities


class GradNormCallback(pl.Callback):

    def on_after_backward(self, trainer, pl_module):
        grad_2norm = pl.utilities.grad_norm(pl_module, norm_type=2.0)[f"grad_2.0_norm_total"]
        pl_module.log("grad_2norm", grad_2norm)
