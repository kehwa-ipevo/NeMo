# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from megatron.core import parallel_state
from omegaconf.omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.multimodal.speech_llm.data.audio_text_dataset import (
    get_audio_text_dataset_from_config,
    get_tarred_audio_text_dataset_from_config,
)
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import TextProcessing
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging, model_utils


class AudioToTextDataModule(pl.LightningDataModule, IOMixin):
    def __init__(self, config: Union[DictConfig, Dict], tokenizer: TokenizerSpec):
        super().__init__()
        self.cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        self.tokenizer = tokenizer
        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None
        self._validation_names = None
        self._test_names = None

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (train/val/test split)
        # called on every process in DDP
        if stage == 'fit' or stage is None:
            self._train_ds = self._create_dataset('train')
            self._validation_ds = self._create_dataset('validation')
        if stage == 'test' or stage is None:
            self._test_ds = self._create_dataset('test')

        if stage != 'predict':
            self.data_sampler = MegatronDataSampler(
                seq_len=self.cfg.max_seq_length,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=self.cfg.global_batch_size,
                rampup_batch_size=self.cfg.get("rampup_batch_size", None),
                dataloader_type="batch",  # "batch" should be used for SFT,
            )

            # Follows the calculation in nemo.collections.nlp.data.language_modeling.megatron.
            # base_dataset_utils.get_datasets_weights_and_num_samples
            self.max_train_samples = int(math.ceil(self.cfg.global_batch_size * self.trainer.max_steps * 1.005))

    @lru_cache
    def _create_dataset(self, mode: str):
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataset creation as it is not specified in the config: {self.cfg}")
            return None

        if 'augmentor' in data_cfg:
            augmentor = process_augmentations(
                data_cfg['augmentor'],
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            augmentor = None

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get("use_lhotse"):
            tp = TextProcessing(
                self.tokenizer,
                max_seq_length=data_cfg["max_seq_length"],
                min_seq_length=data_cfg["min_seq_length"],
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', False),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=data_cfg.get('sep_id', None),
                seed=data_cfg.get('seed', 1234),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=data_cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=data_cfg.get("virtual_tokens", 0),
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                context_key=data_cfg.get('context_key', 'context'),
                answer_key=data_cfg.get('answer_key', 'answer'),
                end_string=data_cfg.get('end_string', None),
                sample_alpha=data_cfg.get('sample_alpha', None),
            )
            return LhotseAudioQuestionAnswerDataset(
                tp,
                default_context="answer the question according to the previous audio",
                tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                max_seq_length=data_cfg["max_seq_length"],
                context_key=data_cfg.get('context_key', "context"),
                default_context_key=data_cfg.get('default_context_key', "default_context"),
            )

        setattr(self, f"_{mode}_names", data_cfg.get('name', None))

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get('is_tarred', False):
            return get_tarred_audio_text_dataset_from_config(
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                sep_id=data_cfg.get('sep_id', None),
                answer_only_loss=data_cfg.get('answer_only_loss', True),
                virtual_tokens=data_cfg.get("virtual_tokens", 0),
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            return get_audio_text_dataset_from_config(
                manifest_filepath=data_cfg.manifest_filepath,
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                is_train=(mode == 'train'),
                sep_id=data_cfg.get('sep_id', None),
                answer_only_loss=data_cfg.get('answer_only_loss', True),
                virtual_tokens=data_cfg.get("virtual_tokens", 0),
            )

    def _create_nemo_dataloader(self, dataset: Any, mode: str, **kwargs) -> DataLoader:
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataloader creation as it is not specified in the config: {self.cfg}")
            return None

        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        elif hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries like ChainDataset
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        if isinstance(dataset, torch.utils.data.IterableDataset):
            data_parallel_size = parallel_state.get_data_parallel_world_size()
            num_micro_batches = data_cfg.global_batch_size // (data_cfg.micro_batch_size * data_parallel_size)
            global_batch_size_on_this_data_parallel_rank = num_micro_batches * data_cfg.micro_batch_size
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                shuffle=False,
                batch_size=global_batch_size_on_this_data_parallel_rank,
                drop_last=True,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
            )
            return dataloader

        return DataLoader(
            dataset,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.get("persistent_workers", False),
            collate_fn=collate_fn,
            **kwargs,
        )

    def _create_lhotse_dataloader(self, dataset: Any, mode: str, **kwargs) -> DataLoader:
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataloader creation as it is not specified in the config: {self.cfg}")
            return None

        if mode == "train":
            return get_lhotse_dataloader_from_config(
                data_cfg,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
                dataset=dataset,
            )
        # for eval, we need to create separate dataset so as to report splitted numbers
        else:
            dls = []
            if hasattr(data_cfg, 'manifest_filepath'):
                manifest_filepath = data_cfg.manifest_filepath
                for cur_manifest_filepath in manifest_filepath:
                    conf = copy.deepcopy(data_cfg)
                    conf['manifest_filepath'] = cur_manifest_filepath
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )
            else:
                input_cfg = data_cfg.input_cfg
                if isinstance(input_cfg, (str, Path)):
                    # Resolve /path/to/input_cfg.yaml into config contents if needed.
                    input_cfg = OmegaConf.load(input_cfg)
                    assert len(input_cfg) == 1, "Only one dataset with multiple manifest paths is supported for eval"
                    data_cfg.input_cfg = input_cfg
                    # for getting names
                    manifest_filepath = [ic.manifest_filepath for ic in input_cfg[0].input_cfg]
                for cur_input_cfg in input_cfg[0].input_cfg:
                    conf = copy.deepcopy(data_cfg)
                    conf.input_cfg[0].input_cfg = [cur_input_cfg]
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )

            if 'name' not in data_cfg:
                names = []
                for cur_manifest_filepath in manifest_filepath:
                    names.append(Path(cur_manifest_filepath).stem)
                OmegaConf.update(data_cfg, 'name', names, force_add=True)
                logging.info(f'Update dataset names as {names}')
            return dls

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_cfg = self.cfg.get("train_ds", None)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._train_ds, 'train')
        else:
            return self._create_nemo_dataloader(self._train_ds, 'train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_cfg = self.cfg.get("validation_ds", None)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._validation_ds, 'validation')
        else:
            if isinstance(self._validation_ds, list):
                if len(self._validation_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'validation') for ds in self._validation_ds]
                else:
                    return self._create_nemo_dataloader(self._validation_ds[0], 'validation')
            else:
                return self._create_nemo_dataloader(self._validation_ds, 'validation')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        data_cfg = self.cfg.get("test_ds", None)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._test_ds, 'test')
        else:
            if isinstance(self._test_ds, list):
                if len(self._test_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'test') for ds in self._test_ds]
                else:
                    return self._create_nemo_dataloader(self._test_ds[0], 'test')
            else:
                return self._create_nemo_dataloader(self._test_ds, 'test')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if "predict_ds" not in self.cfg:
            data_cfg = self.cfg.get("test_ds", None)
            data_key = 'test'
        else:
            data_cfg = self.cfg.get("predict_ds", None)
            data_key = 'predict'

        self._test_ds = self._create_dataset(data_key)
        if data_cfg.get("use_lhotse"):
            return self._create_lhotse_dataloader(self._test_ds, 'predict')
        else:
            if isinstance(self._test_ds, list):
                if len(self._test_ds) > 1:
                    return [self._create_nemo_dataloader(ds, 'predict') for ds in self._test_ds]
                else:
                    return self._create_nemo_dataloader(self._test_ds[0], 'predict')
            else:
                return self._create_nemo_dataloader(self._test_ds, 'predict')

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {'consumed_samples': consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1

    def reconfigure_limit_batches(self):
        # Override limit_train_batches in terms of num of microbatches
        self._reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds, 'train')
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_ds, 'val')

    def _reconfigure_limit_batches(self, limit_batches, dataloader, mode):
        """
        Reconfigure trainer.limit_val_batches for pretraining
        """
        # Override limit_batches in terms of num microbatches and so there are limit_batches//num_micro_batches num of global batches
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        if isinstance(limit_batches, int):
            limit_batches *= get_num_microbatches()
        else:
            assert isinstance(limit_batches, float)
            # Don't reconfigure if limit_batches is 0.0 or if there's no dataloader
            if limit_batches == 0.0 or dataloader is None:
                return
            # len(dataloader) returns len as num of microbatches
            dl_len_in_micro_batches = len(dataloader)
            if len(dataloader) != float("inf"):
                if limit_batches == 1.0:
                    limit_batches = dl_len_in_micro_batches
                else:
                    limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                    if limit_micro_batches == 0 and limit_batches > 0.0:
                        min_percentage = 1.0 / len(dataloader)
                        raise RuntimeError(
                            f"You requested to check {limit_batches} of the val_dataloader but"
                            f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                            f" `limit_val_batches` argument. Try at least"
                            f" `limit_val_batches={min_percentage}`"
                        )
                    # Make sure trainer.limit_val_batches is a multiple of num of microbatches
                    if limit_micro_batches < get_num_microbatches():
                        limit_batches = get_num_microbatches()
                    else:
                        limit_batches = limit_batches - limit_batches % get_num_microbatches()

        if mode == 'train':
            self.trainer.limit_train_batches = limit_batches
        else:
            self.trainer.limit_val_batches = limit_batches

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()