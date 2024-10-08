# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Any, Union, List, Dict, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import get_process_group_ranks
import os
from megatron.core import parallel_state
from typing_extensions import override
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from PIL import Image
import wandb
if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

from nemo.lightning import get_vocab_size, io
from nemo.lightning.pytorch.optim import OptimizerModule
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, get_packed_seq_params
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from tqdm import tqdm
from einops import rearrange

from .dit.dit_model import DiTCrossAttentionModel
from nemo.collections.diffusion.sampler.edm.edm_pipeline import EDMPipeline

def dit_forward_step(model, batch) -> torch.Tensor:
    return model(**batch)

def dit_data_step(module, dataloader_iter):
    # batch = get_batch_on_this_tp_rank(dataloader_iter)
    batch = next(dataloader_iter)[0]
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device='cuda', dtype=torch.bfloat16, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    return batch

def get_batch_on_this_cp_rank(data: Dict):
    """Split the data for context parallelism."""
    from megatron.core import mpu
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    t = 16
    if cp_size > 1:
        # cp split on seq_length, for video_latent, noise_latent and pos_ids
        assert t % cp_size == 0, "t must divisibly by cp_size"
        num_valid_tokens_in_ub = None
        if 'loss_mask' in data and data['loss_mask'] is not None:
            num_valid_tokens_in_ub = data['loss_mask'].sum()

        for key, value in data.items():
            if (value is not None) and (key in ['video', 'video_latent', 'noise_latent', 'pos_ids']):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                B, C, T, H, W = value.shape
                # FIXME packed sequencing
                data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[:, cp_rank, ...].contiguous()
        data['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
    return data

@dataclass
class DiTConfig(TransformerConfig, io.IOMixin):
    """
    Config for DiT-S model
    """
    crossattn_emb_size: int = 1024
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

    num_layers: int = 12
    hidden_size: int = 384
    additional_timestamp_channels: dict = field(default_factory=lambda: dict(fps=64, h=64, w=64, org_h=64, org_w=64))
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 1
    num_attention_heads: int = 6
    layernorm_epsilon=1e-6
    normalization="RMSNorm"
    add_bias_linear=False
    qk_layernorm_per_head=True
    layernorm_zero_centered_gamma=False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    # max_position_embeddings: int = 5400
    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    vae_path: str = None
    pixel_chunk_duration: int =121
    latent_chunk_duration: int = 16
    sigma_data: float = 0.5

    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step


    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        return DiTCrossAttentionModel(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            # position_embedding_type=self.position_embedding_type,
            # rotary_percent=self.rotary_percent,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            additional_timestamp_channels=self.additional_timestamp_channels,
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
        )

    def configure_vae(self):
        from nemo.collections.multimodal.vfm.vae.video_vae import video_vae3_512, AutoencoderKLVAE
        if os.path.exists(os.path.join(self.vae_path, 'config.json')):
            return AutoencoderKLVAE(self.vae_path)
            
        return video_vae3_512(self.vae_path, pixel_chunk_duration=self.pixel_chunk_duration, latent_chunk_duration=self.latent_chunk_duration)
    

@dataclass
class DiTBConfig(DiTConfig):
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12

@dataclass
class DiTLConfig(DiTConfig):
    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    additional_timestamp_channels: dict = field(default_factory=lambda: dict(fps=96, h=96, w=96, org_h=96, org_w=96))    

@dataclass
class DiTXLConfig(DiTConfig):
    num_layers: int = 28
    hidden_size: int = 1152
    num_attention_heads: int = 16
    additional_timestamp_channels: dict = field(default_factory=lambda: dict(fps=96, h=96, w=96, org_h=96, org_w=96))

class DiTModel(GPTModel):
    def __init__(
        self,
        config: Optional[DiTConfig] = None,
        optim: Optional[OptimizerModule] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(
            config or DiTConfig(), optim=optim, model_transform=model_transform
        )
        
        self.vae = None

        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        self.diffusion_pipeline = EDMPipeline(net=self, sigma_data=self.config.sigma_data)

        self._noise_generator = None
        self.seed = 42

        self.vae = None

    def data_step(self, dataloader_iter) -> Dict[str, Any]:
        return self.config.data_step_fn(dataloader_iter)
    
    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
    
    def forward_step(self, batch) -> torch.Tensor:
        data_batch = {k: v.to(device='cuda', non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        if parallel_state.is_pipeline_last_stage():
            output_batch, kendall_loss = self.diffusion_pipeline.training_step(batch, 0)
            if torch.distributed.get_rank() == 0:
                wandb.log({k: output_batch[k] for k in ['edm_loss'] if k in output_batch}, step=self.global_step)
            kendall_loss = torch.mean(kendall_loss, dim=1)
            return kendall_loss
        else:
            output_tensor = self.diffusion_pipeline.training_step(batch, 0)
            return output_tensor 
    
    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def on_validation_start(self):
        if self.vae is None:
            if self.config.vae_path is None:
                warnings.warn('vae_path not specified skipping validation')
                return None
            self.vae = self.config.configure_vae()
        self.vae.to('cuda')

    def on_validation_end(self):
        if self.vae is not None:
            self.vae.to('cpu')

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        state_shape = batch['video'][0].shape
        sample = self.diffusion_pipeline.generate_samples_from_batch(
            batch,
            guidance=7,
            state_shape=state_shape,
            num_steps=35,
            is_negative_prompt=True if 'neg_t5_text_embeddings' in batch else False,
            seed=42,
        )

        video = (1.0 + self.vae.decode(sample  / self.config.sigma_data)).clamp(0, 2) / 2  # [B, 3, T, H, W]

        video = (video * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)    
        
        T = video.shape[2]
        if T == 1:
            image = rearrange(video, 'b c t h w -> (b t h) w c')
            result = image
        else:
            # result = wandb.Video(video, fps=float(batch['fps'])) # (batch, time, channel, height width)
            result = video

        # wandb is on the last rank for megatron, first rank for nemo
        wandb_rank = 0

        if parallel_state.get_data_parallel_src_rank() == wandb_rank:
            if torch.distributed.get_rank() == wandb_rank:
                gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            else:
                gather_list = None
            torch.distributed.gather_object(result, gather_list, wandb_rank, group=parallel_state.get_data_parallel_group())
            if gather_list is not None:
                videos = []
                for video in gather_list:
                    if len(video.shape) == 3:
                        videos.append(wandb.Image(video))
                    else:
                        videos.append(wandb.Video(video, fps=30))
                wandb.log({'prediction': videos}, step=self.global_step)

        return None

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = DummyLossReduction()

        return self._validation_loss_reduction

    def on_validation_model_zero_grad(self) -> None:
        '''
        Small hack to avoid first validation on resume. 
        This will NOT work if the gradient accumulation step should be performed at this point.
        https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False
        
class DummyLossReduction(MegatronLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return torch.tensor(0.0, device=torch.cuda.current_device()), {"avg": torch.tensor(0.0, device=torch.cuda.current_device())}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        return torch.tensor(0.0, device=torch.cuda.current_device())
