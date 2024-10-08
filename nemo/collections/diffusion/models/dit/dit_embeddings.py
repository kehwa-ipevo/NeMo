# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from typing import Optional, Literal, Dict

import math
import numpy as np
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from diffusers.models.embeddings import get_3d_sincos_pos_embed, TimestepEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule

class ParallelTimestepEmbedding(TimestepEmbedding):
    """
    ParallelTimestepEmbedding is a subclass of TimestepEmbedding that initializes
    the embedding layers with an optional random seed for syncronization.

    Args:
        in_channels (int): Number of input channels.
        time_embed_dim (int): Dimension of the time embedding.
        seed (int, optional): Random seed for initializing the embedding layers. 
                              If None, no specific seed is set.

    Attributes:
        linear_1 (nn.Module): First linear layer for the embedding.
        linear_2 (nn.Module): Second linear layer for the embedding.

    Methods:
        __init__(in_channels, time_embed_dim, seed=None): Initializes the embedding layers.
    """
    def __init__(self, 
        in_channels: int,
        time_embed_dim: int,
        seed=None):
        super().__init__(in_channels=in_channels, time_embed_dim=time_embed_dim)
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.linear_1.reset_parameters()
                self.linear_2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embeddings for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C).

        Returns:
            torch.Tensor: Positional embeddings of shape (B, T, H, W, C).
        """
        return super().forward(x.to(torch.bfloat16, non_blocking=True))

def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    """
    Adjusts the positional embeddings tensor to the current context parallel rank.

    Args:
        pos_emb (torch.Tensor): The positional embeddings tensor.
        seq_dim (int): The sequence dimension index in the positional embeddings tensor.

    Returns:
        torch.Tensor: The adjusted positional embeddings tensor for the current context parallel rank.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb
    
class SinCosPosEmb3D(nn.Module):
    """
    SinCosPosEmb3D is a 3D sine-cosine positional embedding module.

    Args:
        model_channels (int): Number of channels in the model.
        len_h (int): Length of the height dimension.
        len_w (int): Length of the width dimension.
        len_t (int): Length of the temporal dimension.
        spatial_interpolation_scale (float, optional): Scale factor for spatial interpolation. Default is 1.0.
        temporal_interpolation_scale (float, optional): Scale factor for temporal interpolation. Default is 1.0.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the positional embeddings for the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (B, T, H, W, C).

            Returns:
                torch.Tensor: Positional embeddings of shape (1, T, H, W, C).
    """
    def __init__(
        self,
        *,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
    ):
        super().__init__()
        param = get_3d_sincos_pos_embed(
            model_channels, [len_h, len_w], len_t, spatial_interpolation_scale, temporal_interpolation_scale
        )
        param = rearrange(param, "(b t) (h w) c -> b c t h w", h=len_h, w=len_w, b=1)
        self.register_buffer("pos_embedding", torch.from_numpy(param).float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        B, C, T, H, W = x.shape
        cp_size = parallel_state.get_context_parallel_world_size()
        embeddings = self.pos_embedding[..., :T * cp_size, :H, :W]
        embeddings = get_pos_emb_on_this_cp_rank(embeddings, seq_dim=2)
        return embeddings

class FactorizedLearnable3DEmbedding(MegatronModule):
    def __init__(
        self,
        config,
        t: int,
        h: int,
        w: int,
        **kwargs,
    ):
        super().__init__(config=config)
        self.emb_t = torch.nn.Embedding(t, config.hidden_size)
        self.emb_h = torch.nn.Embedding(h, config.hidden_size)
        self.emb_w = torch.nn.Embedding(w, config.hidden_size)

        if config.perform_initialization:
            config.init_method(self.emb_t.weight)
            config.init_method(self.emb_h.weight)
            config.init_method(self.emb_w.weight)

    def forward(self, pos_ids: torch.Tensor):
        return self.emb_t(pos_ids[..., 0]) + self.emb_h(pos_ids[..., 1]) + self.emb_w(pos_ids[..., 2])