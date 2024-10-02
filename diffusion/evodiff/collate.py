import numpy as np
import torch

class OAMaskCollaterDNA(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        sequences : list of sequences
        inputs_padded: if inputs are padded (due to truncation in Simple_Collater) set True (default False)

    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
    """
    def __init__(self, mask_idx, pad_idx):
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        # self.tokenizer = tokenizer

    def __call__(self, batch):
        mask_id = self.mask_idx
        pad_id = self.pad_idx
        src=[]
        tgt=[]
        timesteps = []
        masks=[]
        Lmax = 10000
        mask_id = torch.tensor(mask_id, dtype=torch.int64)
        for i, (x, pad_mask, seq_len) in enumerate(batch):
            # Randomly generate timestep and indices to mask
            D = seq_len # D should have the same dimensions as each sequence length
            # if D <= 1:  # for sequence length = 1 in dataset
            #     t = 1
            # else:
            t = np.random.randint(1, D) # randomly sample timestep
            num_mask = (D-t+1) # from OA-ARMS
            # Append timestep
            timesteps.append(num_mask)
            # Generate mask
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask
            index_arr = np.arange(0, Lmax) #index array [1...seq_len]
            mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape) # mask bools indices specified by mask_arr
            # Mask inputs
            mask = torch.tensor(mask, dtype=torch.bool)
            masks.append(mask)
            x_t = ~mask * x + mask * mask_id
            x_t = torch.where(pad_mask, x_t, pad_id)
            src.append(x_t)
            tgt.append(x)

        return (torch.stack(src, dim=0).to(torch.long), torch.tensor(timesteps), torch.stack(tgt, dim=0).to(torch.long), torch.stack(masks, dim=0))