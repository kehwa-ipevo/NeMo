from copy import deepcopy

import torch
from megatron.core import dist_checkpointing
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.slm.data.audio_to_text_data import AudioToTextDataModule
from nemo.collections.slm.models.speech_to_text_llm_model import SpeechToTextLLM, SpeechToTextLLMConfig
from nemo.collections.slm.modules.asr_module import ASRModuleConfig
from nemo.collections.slm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.slm.utils import SpeechToTextLLMPEFT, get_object_list_from_config
from nemo.core.classes.common import Serialization, typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


def speetch_to_text_llm_train(cfg: DictConfig):
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build the model
    tokenizer = AutoTokenizer(cfg['model']['llm']['pretrained_model'])
    model_config = SpeechToTextLLMConfig(
        language_model_class=cfg['model']['llm']['_target_'],
        language_model_config=Serialization.from_config_dict(cfg['model']['llm']['config']),
        speech_model_config=ASRModuleConfig(**cfg['model']['speech_encoder']),
        modality_adapter_config=ModalityAdapterConfig(**cfg['model']['modality_adapter']),
        language_model_from_pretrained=cfg['model']['llm']['pretrained_model'],
        freeze_language_model=cfg['model']['freeze_language_model'],
        freeze_speech_model=cfg['model']['freeze_speech_model'],
        freeze_modality_adapter=cfg['model']['freeze_modality_adapter'],
    )

    model = SpeechToTextLLM(config=model_config, tokenizer=tokenizer)

    # 2. build dataset
    data = AudioToTextDataModule(cfg['data'], tokenizer=tokenizer)

    # 3. setup the optimizer
    optim = Serialization.from_config_dict(cfg['optim'])

    # 4. setup trainer
    trainer = nl.Trainer(
        strategy=Serialization.from_config_dict(cfg['strategy']),
        plugins=get_object_list_from_config(cfg['plugins']),
        callbacks=get_object_list_from_config(cfg['callbacks']),
        **cfg['trainer'],
    )

    # 5. setup PEFT
    peft = None
    if cfg['model'].get('peft', None):
        peft = SpeechToTextLLMPEFT(peft=Serialization.from_config_dict(cfg['model']['peft']))
        # if model_config.language_model_from_pretrained is None:
        #     raise ValueError("PEFT requires a pretrained model to be specified in the model config.")
        # if 'restore_config' not in cfg['resume']:
        #     ckpt_path = model_config.language_model_from_pretrained
        #     if not dist_checkpointing.check_is_distributed_checkpoint(ckpt_path):
        #         llm_model_cls = model_utils.import_class_by_path(model_config.language_model_class)  # type: GPTModel
        #         ckpt_path = llm_model_cls.import_ckpt(f"{model_config.language_model_hub}{model_config.language_model_from_pretrained}")
        #     restore_cfg = {
        #         '_target_': 'nemo.lightning.RestoreConfig',
        #         'path': ckpt_path, #f"{model_config.language_model_hub}{model_config.language_model_from_pretrained}",
        #     }
        #     cfg['resume']['restore_config'] = restore_cfg

    # 6. setup logger and auto-resume
    resume = Serialization.from_config_dict(cfg['resume'])
    logger = Serialization.from_config_dict(cfg['logger'])

    # 7. train the model
    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
        log=logger,
        peft=peft,
        resume=resume,
    )
    return logger.log_dir