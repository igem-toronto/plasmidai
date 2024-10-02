# Copied from https://github.com/microsoft/evodiff/blob/main/evodiff/losses.py

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
# from evodiff.utils import Tokenizer
from sequence_models.constants import MSA_AAS


class OAMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True):
        self.reweight=reweight
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps, input_mask):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
            input_mask = input_mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        input_mask = input_mask.bool() # padded seq
        # Select
        mask_tokens = mask.sum() # masked tokens
        nonpad_tokens = input_mask.sum(dim=1) # nonpad tokens
        p = torch.masked_select(pred, mask).view(mask_tokens, -1) # [T x K] predictions for each mask char
        t = torch.masked_select(tgt, mask.squeeze()) # [ T ] true mask char
        loss = super().forward(p, t) # [ T ] loss per mask char
        # Calculate reweighted CE loss and NLL loss
        nll_losses = loss.mean()
        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            rwt_term = 1. / timesteps
            rwt_term = rwt_term.repeat_interleave(timesteps)
            _n_tokens = nonpad_tokens.repeat_interleave(timesteps)
            ce_loss = _n_tokens * rwt_term * loss
            ce_losses = ce_loss.mean()  # reduce mean
        else:
            ce_losses = nll_losses
        return ce_losses, nll_losses.to(torch.float64)

