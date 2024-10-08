# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.multimodal.vfm.data.energon_crude_dataset import DiffusionTaskEncoder
import torch

from megatron.core.optimizer import OptimizerConfig
import pytorch_lightning as pl

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.models.model import DiTLConfig, DiTModel, DiTConfig, DiTSConfig, DiTXLConfig
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
import nemo_run as run
from pytorch_lightning.loggers import WandbLogger
from nemo.lightning.pytorch.callbacks import PreemptionCallback


@run.cli.factory
@run.autoconvert
def multimodal_datamodule() -> pl.LightningDataModule:
    data_module = DiffusionDataModule(
        seq_length=21760,
        task_encoder=run.Config(DiffusionTaskEncoder),
        micro_batch_size=1,
        global_batch_size=32,
    )
    return data_module

@run.cli.factory
@run.autoconvert
def peft(args) -> ModelTransform:
    return llm.peft.LoRA(
            target_modules=['linear_qkv', 'linear_proj'], #, 'linear_fc1', 'linear_fc2'],
            dim=args.lora_dim,
        )

@run.cli.factory(target=llm.train)
def pretrain() -> run.Partial:
    return run.Partial(
        llm.train,
        model=run.Config(
            DiTModel,
            config=run.Config(DiTConfig), 
        ),
        data=multimodal_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=0,
            max_epochs=10000,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    ModelCheckpoint,
                    monitor='reduced_train_loss',
                    filename='{epoch}-{step}',
                    every_n_train_steps=1000,
                    save_top_k=-1,
                ),
                run.Config(PreemptionCallback),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger(project='vfm') if "WANDB_API_KEY" in os.environ else None)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=5e-5,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                # weight_decay=0.3,
            )
        ),
        tokenizer=None,
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )

@run.cli.factory(target=llm.train)
def pretrain_xl() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiTXLConfig)
    return recipe

@run.cli.factory(target=llm.train)
def pretrain_l() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiTLConfig)
    return recipe

@run.cli.factory(target=llm.train)
def pretrain_long() -> run.Partial:
    recipe = pretrain()
    # recipe.optim.config.lr = 1e-6
    # recipe.data = videofolder_datamodule()
    recipe.model.config = run.Config(DiTConfig)

    # recipe.trainer.max_steps=1000
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.context_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True

    recipe.optim.lr_scheduler = run.Config(nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=100, hold_steps=1e9)

    recipe.resume.restore_config = run.Config(
        RestoreConfig
    )
    
    return recipe

@run.cli.factory(target=llm.train)
def pretrain_image() -> run.Partial:
    recipe = pretrain()

    recipe.model.config.max_img_h=160
    recipe.model.config.max_img_w=160
    recipe.optim.config.lr=1e-4
    recipe.optim.config.weight_decay=0
    recipe.trainer.limit_val_batches=1
    recipe.trainer.val_check_interval=1000
    # recipe.trainer.num_sanity_val_steps=1
    # recipe.trainer.callbacks[0].every_n_train_steps=1 
    # recipe.trainer.callbacks[0].save_top_k=10
    recipe.data.global_batch_size=256
    recipe.data.use_train_split_for_val=True
    recipe.data.num_workers=8

    recipe.optim.lr_scheduler = run.Config(nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=100, hold_steps=1e9)    
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_image_s_oss() -> run.Partial:
    recipe = pretrain_image()
    recipe.model=run.Config(
        DiTModel,
        config=run.Config(DiTSConfig, max_img_h=160, max_img_w=160, max_frames=2), 
    )
    return recipe

@run.cli.factory(target=llm.train)
def dreambooth() -> run.Partial:
    recipe = pretrain()
    recipe.optim.config.lr = 1e-6
    recipe.data = multimodal_datamodule()
    recipe.model.config = run.Config(DiTConfig)

    recipe.trainer.max_steps=1000
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True

    recipe.resume.restore_config = run.Config(
        RestoreConfig
    )
    recipe.resume.resume_if_exists = False
    
    return recipe

if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=dreambooth)
