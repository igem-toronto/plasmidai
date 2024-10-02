# Copied from https://github.com/microsoft/evodiff/blob/main/evodiff/model.py
from functools import partial
import json
import os
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock
from sequence_models.constants import MSA_PAD, MASK, MSA_ALPHABET
from esm.modules import LearnedPositionalEmbedding, RobertaLMHead, AxialTransformerLayer
from mamba_ssm.models.mixer_seq_simple import Mamba, MambaConfig, _init_weights, load_config_hf, load_state_dict_hf;
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)

class PositionalEncoding(nn.Module):

    """
    2D Positional encoding for transformer
    :param d_model: dimension of the model
    :param max_len: max number of positions
    """

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x.reshape(x.shape[1], x.shape[0], x.shape[2]) # [b x l x e]

class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True,
                 timesteps=None):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps) # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            # expand dim of e2 to match e1
            e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1])
            e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])
            e = torch.add(e2, e)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)



class MSATransformerTime(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"
    https://doi.org/10.1101/2021.02.12.430858
    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, use_ckpt=False, n_tokens=len(MSA_ALPHABET),
                 padding_idx=MSA_ALPHABET.index(MSA_PAD), mask_idx=MSA_ALPHABET.index(MASK),
                 max_positions=1024, timesteps=None):
        super(MSATransformerTime, self).__init__()

        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_model, timesteps) # Timestep encoding
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        # self.contact_head = ContactPredictionHead()
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )

        self.use_ckpt = use_ckpt

    def forward(self, tokens, timesteps):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        #print("tokens", tokens.shape) # B, D, L (batch, depth length)
        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        #print("x", x.shape) # B, D, L, E
        y = self.time_encoding(timesteps)
        y = y.unsqueeze(1).unsqueeze(1)
        y = y.expand(y.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x += y

        # ADD 1 to query sequence in MSA (encode query sequence)
        q = torch.zeros(x.shape)
        q = q.to(x.device)
        q[:,0,:,0] += 1 # add encoding to 1st sequence (query seq) in MSA
        x += q
        #

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x

        # Most of this code is slightly modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
# and https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py





class BiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.fwd_mamba = Mamba(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max,
                            dt_init, dt_scale, dt_init_floor, conv_bias, bias,
                            use_fast_path, layer_idx, device, dtype)
        
        self.rev_mamba = Mamba(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max,
                            dt_init, dt_scale, dt_init_floor, conv_bias, bias,
                            use_fast_path, layer_idx, device, dtype)
        
        self.rev_mamba.in_proj.weight = self.fwd_mamba.in_proj.weight
        self.rev_mamba.in_proj.bias = self.fwd_mamba.in_proj.bias
        self.rev_mamba.out_proj.weight = self.fwd_mamba.out_proj.weight
        self.rev_mamba.out_proj.bias = self.fwd_mamba.out_proj.bias


    def forward(self, x, input_mask, inference_params=None):
        x = x * input_mask
        return self.fwd_mamba(x) + self.rev_mamba(x.flip(dims=[-1])) 

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(BiMambaBlock, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, input_mask = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, input_mask=input_mask, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, input_mask, inference_params=None):
        hidden_states = self.embedding(input_ids * input_mask.int())
        residual = None
        input_mask = input_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, input_mask, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states
    
class BiMamba(nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        # if self.config.tie_embeddings:
        self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, input_mask, inference_params=None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, input_mask, inference_params=inference_params)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)

