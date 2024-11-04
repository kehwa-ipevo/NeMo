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

import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import nemo_run as run
import pytorch_lightning as pl
import torch
from rich.console import Console
from typing_extensions import Annotated

import nemo.lightning as nl
from nemo.lightning import AutoResume, NeMoLogger, OptimizerModule, Trainer, io
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.pytorch.callbacks import PEFT, ModelTransform
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from megatron.core.inference.inference_request import InferenceRequest


TokenizerType = Any


@run.cli.entrypoint(namespace="llm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
    # TODO: Fix export export: Optional[str] = None,
) -> Path:
    """
    Trains a model using the specified data and trainer, with optional tokenizer, source, and export.

    Args:
        model (pl.LightningModule): The model to be trained.
        data (pl.LightningDataModule): The data module containing training data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[Union[AutoResume, Resume]]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[TokenizerType]): Tokenizer setting to be applied. Can be 'data' or 'model' or an instance of TokenizerSpec.
        export (Optional[str]): Filename to save the exported checkpoint after training.
        model_transform (Optional[Union[Callable[[nn.Module], nn.Module], PEFT]]): A model transform to be applied.

    Returns
    -------
        Path: The directory path where training artifacts are saved.

    Examples
    --------
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> train(model, data, trainer, tokenizer="data")
        PosixPath('/path/to/log_dir')
    """
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )

    trainer.fit(model, data)

    return app_state.exp_dir


@run.cli.entrypoint(namespace="llm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
) -> Path:
    """
    Pretrains a model using the specified data and trainer, with optional logging, resuming, and optimization.

    This function is a wrapper around the `train` function, specifically configured for pretraining tasks.
    Note, by default it will use the tokenizer from the model.

    Args:
        model (pl.LightningModule): The model to be pretrained.
        data (pl.LightningDataModule): The data module containing pretraining data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default
            optimizer from the model will be used.

    Returns:
        Path: The directory path where pretraining artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.PretrainingDataModule(paths=[...], seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> llm.pretrain(model, data, trainer)
        PosixPath('/path/to/log_dir')
    """
    _validate_config(model, data, trainer, log=log, resume=resume, optim=optim)
    return train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer="data",
    )


@run.cli.entrypoint(namespace="llm")
def finetune(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    peft: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Finetunes a model using the specified data and trainer, with optional logging, resuming, and PEFT.

    Note, by default it will use the tokenizer from the model.

    Args:
        model (pl.LightningModule): The model to be finetuned.
        data (pl.LightningDataModule): The data module containing finetuning data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default
            optimizer from the model will be used.
        peft (Optional[PEFT]): A PEFT (Parameter-Efficient Fine-Tuning) configuration to be applied.

    Returns:
        Path: The directory path where finetuning artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> finetune(model, data, trainer, peft=llm.peft.LoRA()])
        PosixPath('/path/to/log_dir')
    """

    _validate_config(model, data, trainer, log=log, resume=resume, optim=optim, model_transform=peft)
    return train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer="model",
        model_transform=peft,
    )


@run.cli.entrypoint(namespace="llm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Validates a model using the specified data and trainer, with optional logging, resuming, and model transformations.

    Args:
        model (pl.LightningModule): The model to be validated.
        data (pl.LightningDataModule): The data module containing validation data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[AutoResume]): Resume from a checkpoint for validation.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[TokenizerType]): Tokenizer setting to be applied. Can be 'data' or 'model' or an instance of TokenizerSpec.
        model_transform (Optional[Union[Callable[[nn.Module], nn.Module], PEFT]]): A model transform to be applied.

    Returns:
        Path: The directory path where validation artifacts are saved.

    Examples:
        >>> from nemo.collections import llm
        >>> from nemo import lightning as nl
        >>> model = llm.MistralModel()
        >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
        >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
        >>> validate(model, data, trainer, tokenizer="data")
        PosixPath('/path/to/log_dir')
    """
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )

    trainer.validate(model, data)

    return app_state.exp_dir


def get_trtllm_deployable(
    nemo_checkpoint,
    model_type,
    triton_model_repository,
    num_gpus,
    tensor_parallelism_size,
    pipeline_parallelism_size,
    max_input_len,
    max_output_len,
    max_batch_size,
    dtype,
):
    from nemo.export.tensorrt_llm import TensorRTLLM

    if triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = triton_model_repository

    if nemo_checkpoint is None and triton_model_repository is None:
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a TensorRT-LLM engine."
        )

    if nemo_checkpoint is None and not os.path.isdir(triton_model_repository):
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a valid TensorRT-LLM engine."
        )

    if nemo_checkpoint is not None and model_type is None:
        raise ValueError("Model type is required to be defined if a nemo checkpoint is provided.")

    trt_llm_exporter = TensorRTLLM(
        model_dir=trt_llm_path,
        load_model=(nemo_checkpoint is None),
    )

    if nemo_checkpoint is not None:
        try:
            logging.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=nemo_checkpoint,
                model_type=model_type,
                n_gpus=num_gpus,
                tensor_parallelism_size=tensor_parallelism_size,
                pipeline_parallelism_size=pipeline_parallelism_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                dtype=dtype,
            )
        except Exception as error:
            raise RuntimeError("An error has occurred during the model export. Error message: " + str(error))

    return trt_llm_exporter


def store_args_to_json(triton_http_address, triton_port, triton_request_timeout, openai_format_response):
    args_dict = {
        "triton_service_ip": triton_http_address,
        "triton_service_port": triton_port,
        "triton_request_timeout": triton_request_timeout,
        "openai_format_response": openai_format_response,
    }
    with open("nemo/deploy/service/config.json", "w") as f:
        json.dump(args_dict, f)


@run.cli.entrypoint(namespace="llm")
def deploy(
    nemo_checkpoint: Path = None,
    model_type: str = "llama",
    triton_model_name: str = "xxx",
    triton_model_version: Optional[int] = 1,
    triton_port: int = 8080,
    triton_http_address: str = "0.0.0.0",
    triton_request_timeout: int = 60,
    triton_model_repository: Path = None,
    num_gpus: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    dtype: str = "bfloat16",
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    start_rest_service: bool = False,
    rest_service_http_address: str = "0.0.0.0",
    rest_service_port: int = 8000,
    openai_format_response: bool = False,
):
    from nemo.deploy import DeployPyTriton

    if start_rest_service:
        if triton_port == rest_service_port:
            logging.error("REST service port and Triton server port cannot use the same port.")
            return
        # Store triton ip, port and other args relevant for REST API in config.json to be accessible by rest_model_api.py
        store_args_to_json(triton_http_address, triton_port, triton_request_timeout, openai_format_response)

    triton_deployable = get_trtllm_deployable(
        nemo_checkpoint,
        model_type,
        triton_model_repository,
        num_gpus,
        tensor_parallelism_size,
        pipeline_parallelism_size,
        max_input_len,
        max_output_len,
        max_batch_size,
        dtype,
    )

    try:
        nm = DeployPyTriton(
            model=triton_deployable,
            triton_model_name=triton_model_name,
            triton_model_version=triton_model_version,
            max_batch_size=max_batch_size,
            port=triton_port,
            address=triton_http_address,
        )

        logging.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        logging.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    uvicorn_supported = True
    try:
        import uvicorn
    except ImportError as error:
        logging.warning(f"uvicorn could not be imported: {error}")
        uvicorn_supported = False

    try:
        logging.info("Model serving on Triton is will be started.")
        if start_rest_service and uvicorn_supported:
            try:
                logging.info("REST service will be started.")
                uvicorn.run(
                    "nemo.deploy.service.rest_model_api:app",
                    host=rest_service_http_address,
                    port=rest_service_port,
                    reload=True,
                )
            except Exception as error:
                logging.error("Error message has occurred during REST service start. Error message: " + str(error))
        nm.serve()
    except Exception as error:
        logging.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    logging.info("Model serving will be stopped.")
    nm.stop()


@run.cli.entrypoint(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Imports a checkpoint into a model using the model's associated importer, typically for
    the purpose of fine-tuning a community model trained in an external framework, such as
    Hugging Face.

    This function can be used both programmatically and through the NeMo CLI:

    CLI Usage:
    ```bash
    # Import Llama 3 8B from HuggingFace (saves to $NEMO_MODELS_CACHE)
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B"

    # Import with custom output path
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B" output_path="/path/to/save"

    # Force overwrite existing checkpoint
    nemo llm import llama3_8b source="hf://meta-llama/Llama-3.1-8B" overwrite=true
    ```

    Python Usage:
    ```python
    model = Mistral7BModel()
    imported_path = import_ckpt(model, "hf://mistralai/Mistral-7B-v0.1")
    ```

    The importer component of the model reads the checkpoint data from the specified source
    and transforms it into the right format. This is particularly useful for adapting
    models that have been pre-trained in different environments or frameworks to be fine-tuned
    or further developed within the current system.

    For instance, using `import_ckpt(Mistral7BModel(), "hf")` initiates the import process
    by searching for a registered model importer tagged with "hf". In NeMo, `HFMistral7BImporter`
    is registered under this tag via:
    `@io.model_importer(Mistral7BModel, "hf", default_path="mistralai/Mistral-7B-v0.1")`.
    This links `Mistral7BModel` to `HFMistral7BImporter`, designed for HuggingFace checkpoints.

    Args:
        model (pl.LightningModule): The model into which the checkpoint will be imported.
            This model must implement the ConnectorMixin.
        source (str): The source from which the checkpoint will be imported. This can be
            a file path, URL, or any other string identifier that the model's importer
            can recognize.
        output_path (Optional[Path]): The path where the imported checkpoint will be stored.
            If not specified, the checkpoint will be saved to $NEMO_MODELS_CACHE
            (defaults to ~/.cache/nemo/models/ if the environment variable is not set).
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.

    Returns:
        Path: The path where the checkpoint has been saved after import.

    Raises:
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary importer functionality.
    """
    output = io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite)

    console = Console()
    if output_path:
        console.print(f"[green]✓ Checkpoint imported to {output}[/green]")
    else:
        console.print(f"[green] $NEMO_MODELS_CACHE={NEMO_MODELS_CACHE} [/green]")
        console.print(f"[green]✓ Checkpoint imported to {output}[/green]")

    return output


def load_connector_from_trainer_ckpt(path: Path, target: str) -> io.ModelConnector:
    return io.load_context(path).model.exporter(target, path)


@run.cli.entrypoint(name="export", namespace="llm")
def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
) -> Path:
    """
    Exports a checkpoint from a model using the model's associated exporter, typically for
    the purpose of sharing a model that has been fine-tuned or customized within NeMo.

    This function can be used both programmatically and through the NeMo CLI:

    CLI Usage:
    ```bash
    # Export model to HuggingFace format (saves to {checkpoint_path}/hf/)
    nemo llm export /path/to/model.nemo target="hf"

    # Export with custom output path
    nemo llm export /path/to/model.nemo target="hf" output_path="/path/to/save"

    # Force overwrite existing export
    nemo llm export /path/to/model.nemo target="hf" overwrite=true
    ```

    Python Usage:
    ```python
    nemo_ckpt_path = Path("/path/to/model.nemo")
    export_path = export_ckpt(nemo_ckpt_path, "hf")
    ```

    The exporter component of the model reads the model's state from the specified path and
    exports it into the format specified by the 'target' identifier. This is particularly
    useful for adapting models that have been developed or fine-tuned within NeMo to be
    compatible with other environments or frameworks.

    Args:
        path (Path): The path to the model's checkpoint file from which data will be exported.
        target (str): The identifier for the exporter that defines the format of the export
            (e.g., "hf" for HuggingFace format).
        output_path (Optional[Path]): The path where the exported checkpoint will be saved.
            If not specified, defaults to {checkpoint_path}/{target}/.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.
        load_connector (Callable[[Path, str], ModelConnector]): A function to load the appropriate
            exporter based on the model and target format. Defaults to `load_connector_from_trainer_ckpt`.

    Returns:
        Path: The path where the checkpoint has been saved after export.

    Raises:
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary exporter functionality.
    """
    output = io.export_ckpt(path, target, output_path, overwrite, load_connector)

    console = Console()
    console.print(f"[green]✓ Checkpoint exported to {output}[/green]")

    return output


@run.cli.entrypoint(name="generate", namespace="llm")
def generate(
    path: Union[Path, str],
    prompts: list[str],
    trainer: nl.Trainer,
    encoder_prompts: Optional[list[str]] = None,
    params_dtype: torch.dtype = torch.bfloat16,
    add_BOS: bool = False,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_params: Optional["CommonInferenceParams"] = None,
    text_only: bool = False,
) -> list[Union["InferenceRequest", str]]:
    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=path,
        trainer=trainer,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
    )
    results = inference.generate(
        model=inference_wrapped_model,
        tokenizer=mcore_tokenizer,
        prompts=prompts,
        encoder_prompts=encoder_prompts,
        add_BOS=add_BOS,
        max_batch_size=max_batch_size,
        random_seed=random_seed,
        inference_params=inference_params,
    )

    return [r.generated_text if text_only else r for r in results]


def _use_tokenizer(model: pl.LightningModule, data: pl.LightningDataModule, tokenizer: TokenizerType) -> None:
    if tokenizer == "data":
        _set_with_io(model, "tokenizer", data.tokenizer)
    elif tokenizer == "model":
        _set_with_io(data, "tokenizer", model.tokenizer)
    else:
        try:
            from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

            if isinstance(tokenizer, TokenizerSpec):
                _set_with_io(model, "tokenizer", tokenizer)
                _set_with_io(data, "tokenizer", tokenizer)
            else:
                raise ValueError(f"Expected TokenizerSpec or 'data' or 'model', got: {tokenizer}")
        except ImportError:
            raise ValueError("TokenizerSpec is not available")


def _setup(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Optional[NeMoLogger],
    resume: Optional[AutoResume],
    optim: Optional[OptimizerModule],
    tokenizer: Optional[TokenizerType],
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]],
) -> Any:  # Return type is Any because app_state's type is not specified
    _log = log or NeMoLogger()
    if resume and isinstance(model_transform, PEFT) and _log.ckpt:
        logging.info("Disabling try_restore_best_ckpt restoration for adapters")
        _log.ckpt.try_restore_best_ckpt = False

    app_state = _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=getattr(train, "__io__", None),
    )
    if resume is not None:
        resume.setup(trainer, model)

    if optim:
        optim.connect(model)
    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)

    if model_transform:
        _set_with_io(model, "model_transform", model_transform)

    # Add ModelTransform callback to Trainer if needed
    if getattr(model, "model_transform", None):
        if not any(isinstance(cb, ModelTransform) for cb in trainer.callbacks):
            if isinstance(model_transform, ModelTransform):
                trainer.callbacks.append(model_transform)
            else:
                trainer.callbacks.append(ModelTransform())

    return app_state


def _set_with_io(obj, attr, value):
    setattr(obj, attr, value)
    if hasattr(obj, "__io__") and hasattr(value, "__io__"):
        setattr(obj.__io__, attr, deepcopy(value.__io__))


def _validate_config(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Optional[NeMoLogger] = None,
    resume: Optional[AutoResume] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> None:

    ## Model validation
    assert model.config.seq_length > 0
    assert model.config.num_layers > 0
    assert model.config.hidden_size > 0
    assert model.config.num_attention_heads > 0
    assert model.config.ffn_hidden_size > 0

    if model.config.seq_length is not None:
        if model.config.max_position_embeddings is not None:
            assert model.config.seq_length <= model.config.max_position_embeddings

    ## Data validation
    assert data.config.micro_batch_size > 0
    assert data.config.global_batch_size > 0
    assert data.config.seq_length > 0

    assert (
        data.config.global_batch_size % data.config.micro_batch_size == 0
    ), "Global batch size must be divisible by micro batch size."

    ## Parallelism validation
    assert isinstance(trainer.strategy, run.Config)

    # DP size validation
    assert (trainer.num_nodes * trainer.devices) % (
        trainer.strategy.tensor_model_parallel_size
        * trainer.strategy.pipeline_model_parallel_size
        * trainer.strategy.context_parallel_size
    ) == 0, "Number of GPUs must be divisible by the product of all parallelism sizes for data parallel."

    assert (
        data.config.global_batch_size
        % (
            data.config.micro_batch_size
            * (
                (trainer.num_nodes * trainer.devices)
                / (
                    trainer.strategy.tensor_model_parallel_size
                    * trainer.strategy.pipeline_model_parallel_size
                    * trainer.strategy.context_parallel_size
                )
            )
        )
        == 0
    )

    # TP/SP validation
    if trainer.strategy.tensor_model_parallel_size == 1:
        if trainer.strategy.pipeline_model_parallel_size != 1:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
            trainer.strategy.sequence_parallel = False

    # PP/VP validation
    if trainer.strategy.pipeline_model_parallel_size > 1:
        assert (
            trainer.strategy.pipeline_dtype is not None
        ), "pipeline_dtype must be set if pipeline model parallelism is enabled"
    else:
        if trainer.strategy.virtual_pipeline_parallel_size is not None:
            warnings.warn("Disabling virtual pipeline parallelism because pipeline model parallelism is disabled")
            trainer.strategy.virtual_pipeline_parallel_size = None
        if trainer.strategy.pipeline_dtype is not None:
            warnings.warn("Setting pipeline dtype to None because pipeline model parallelism is disabled")
            trainer.strategy.pipeline_dtype = None

    # CP validation
    if trainer.strategy.context_parallel_size > 1:
        if model.config.seq_length is not None:
            assert model.config.seq_length % (trainer.strategy.context_parallel_size * 2) == 0, (
                'seq-length should be a multiple of 2 * context-parallel-size ' 'if context-parallel-size > 1.'
            )

    # EP validation
    if trainer.strategy.expert_model_parallel_size > 1:
        assert model.config.num_moe_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert (
            model.config.num_moe_experts % trainer.strategy.expert_model_parallel_size == 0
        ), "Number of experts should be a multiple of expert model parallel_size."
