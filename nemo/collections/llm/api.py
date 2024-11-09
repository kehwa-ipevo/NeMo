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


@run.cli.entrypoint(namespace="llm")
def deploy(
    nemo_checkpoint: Path = None,
    model_type: str = "llama",
    triton_model_name: str = 'triton_model',
    triton_model_version: Optional[int] = 1,
    triton_port: int = 8000,
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
    start_rest_service: bool = True,
    rest_service_http_address: str = "0.0.0.0",
    rest_service_port: int = 8080,
    openai_format_response: bool = True,
    output_generation_logits: bool = True
):
    """
    Deploys nemo model on a PyTriton server by converting the nemo ckpt to trtllm.
    Also starts rest service that is used to send OpenAI API compatible input request
    to the PyTiton server.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        model_type (str): Type of the model. Choices: gpt, llama, falcon, starcoder. Default: llama.
        triton_model_name (str): Name for the model that gets deployed on PyTriton. Please ensure that the same model name
        is passed to the evalute method for the model to be accessible while sending evalution requests.  Default: 'triton_model'.
        triton_model_version (Optional[int]): Version for the triton model. Default: 1.
        triton_port (int): Port for the PyTriton server. Default: 8000.
        triton_http_address (str): HTTP address for the PyTriton server. Default:  "0.0.0.0".
        triton_request_timeout (int): Timeout in seconds for Triton server. Default: 60,
        triton_model_repository (Path): Folder for the trt-llm conversion, trt-llm engin gets saved in this path specified. Default: None.
        num_gpus (int): Number of GPUs for export to trtllm and deploy. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        dtype (str): dtype of the TensorRT-LLM model. Default: "bfloat16".
        max_input_len (int): Max input length of the model. Default: 256.
        max_output_len (int): Max output length of the model. Default: 256.
        max_batch_size (int): Max batch size of the model. Default: 8.
        start_rest_service (bool): Start rest service that is used to send evaluation requests to the PyTriton server. Needs to be True
        to be able to run evaluation . Default: True.
        rest_service_http_address (str): HTTP address for the rest service. Default: "0.0.0.0".
        rest_service_port (int): Port for the rest service. Ensure the rest service port is the port fowarded between host machine and docker
        when running locally inside a docker container. Default: 8080.
        openai_format_response (bool): Return the response from PyTriton server in OpenAI compatible format. Needs to be True while running evaluation.
        Default: True.
        output_generation_logits (bool): If true builds trtllm engine with gather_generation_logits set to True. generation_logits are used to compute the
        logProb of the output token. Default: True.
    """
    from nemo.deploy import DeployPyTriton
    from nemo.collections.llm import evaluation

    evaluation.unset_environment_variables()
    if start_rest_service:
        if triton_port == rest_service_port:
            logging.error("REST service port and Triton server port cannot use the same port.")
            return
        # Store triton ip, port and other args relevant for REST API as env vars to be accessible by rest_model_api.py
        os.environ['TRITON_HTTP_ADDRESS'] = triton_http_address
        os.environ['TRITON_PORT'] = str(triton_port)
        os.environ['TRITON_REQUEST_TIMEOUT'] = str(triton_request_timeout)
        os.environ['OPENAI_FORMAT_RESPONSE'] = str(openai_format_response)
        os.environ['OUTPUT_GENERATION_LOGITS'] = str(output_generation_logits)

    triton_deployable = evaluation.get_trtllm_deployable(
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
        output_generation_logits
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
        nm.run()
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


def evaluate(
    nemo_checkpoint_path: Path,
    url: str = "http://0.0.0.0:1234/v1",
    model_name: str = "triton_model",
    eval_task: str = "gsm8k",
    num_fewshot: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    # inference params
    max_tokens_to_generate: Optional[int] = 256,
    temperature: Optional[float] = 0.000000001,
    top_p: Optional[float] = 0.0,
    top_k: Optional[int] = 1,
    add_bos: Optional[bool] = False,
):
    """
    Evaluates nemo model deployed on PyTriton server (via trtllm) using lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness/tree/main).
    nemo_checkpoint_path (Path): Path for nemo 2.0 checkpoint. This is used to get the tokenizer from the ckpt which is
    required to tokenize the evaluation input and output prompts.
    url (str): rest serice url and port that were used in the deploy method above in the format: http://{rest_service_http}:{rest_service_port}.
    Post requests with evaluation input prompts (from lm-eval-harness) are sent to this url which is then passed to the model deployed in PyTriton server.
    The rest service url and port serve as the entry point to evaluate model deployed on PyTriton server.
    model_name (str): Name of the model that is deployed on PyTriton server. It should be the same as triton_model_name passed to the deploy method above to be able
    to launch evaluation.
    eval_task (str): task to be evaluated on. For ex: "gsm8k", "gsm8k_cot", "mmlu", "lambada". Default: "gsm8k".
    These are the tasks that are supported currently. Any other task of type generate_until or loglikelihood from lm-evaluation-harness can be run,
    but only the above mentioned ones are tested. Tasks of type loglikelihood_rolling are not supported yet.
    num_fewshot (int): number of examples in few-shot context. Default: None.
    limit (Union[int, float]): Limit the number of examples per task. If <1 (i.e float val between 0 and 1), limit is a percentage of the total number of examples.
    If int say x, then run evaluation only on x number of samples/samples from the eval dataset. Default: None, which means eval is run the entire dataset.
    bootstrap_iters (int): Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed. Default: 100000.
    # inference params
    max_tokens_to_generate (int): max tokens to generate. Default: 256.
    temperature: Optional[float]: float value between 0 and 1. temp of 0 indicates greedy decoding, where the token with highest prob is chosen. Default: 0.000000001.
    Temp can't be set to 0.0, due to a bug with TRTLLM (# TODO to be investigated) hence using a very samll value.
    top_p: Optional[float]: float value between 0 and 1. limits to the top tokens within a certain probability. top_p=0 means the model will only consider
    the single most likely token for the next prediction. Default: 0.0.
    top_k: Optional[int]: limits to a certain number (K) of the top tokens to consider. top_k=1 means the model will only consider the single most likely token
    for the next prediction. Default: 1
    add_bos: Optional[bool]: whether a special token representing the beginning of a sequence should be added when encoding a string. Default: False since typically for
    CausalLM its set to False. If needed set add_bos to True.

    """
    try:
        # lm-evaluation-harness import
        from lm_eval import evaluator
    except ImportError:
        raise ImportError("Please ensure that lm-evaluation-harness is installed in your env as it is required to run evaluations")

    from nemo.collections.llm import evaluation

    # Get tokenizer from nemo ckpt. This works only with NeMo 2.0 ckpt.
    tokenizer = io.load_context(nemo_checkpoint_path + '/context', subpath="model").tokenizer
    # Wait for rest service to be ready before starting evaluation
    evaluation.wait_for_rest_service(rest_url=f"{url}/v1/health")
    # Create an object of the NeMoFWLM which is passed as a model to evaluator.simple_evaluate
    model = evaluation.NeMoFWLMEval(model_name, url, tokenizer, max_tokens_to_generate, temperature, top_p, top_k, add_bos)
    results = evaluator.simple_evaluate(
        model=model, tasks=eval_task, limit=limit, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters
    )

    print("score", results['results'][eval_task])


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
    """
    Generates text using a NeMo LLM model.

    This function takes a checkpoint path and a list of prompts,
    and generates text based on the loaded model and parameters.
    It returns a list of generated text, either as a string or as an InferenceRequest object.

    Python Usage:
    ```python
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    prompts = [
        "Hello, how are you?",
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
    ]

    if __name__ == "__main__":
        results = api.generate(
            path=os.path.join(os.environ["NEMO_HOME"], "models", "meta-llama/Meta-Llama-3-8B"),
            prompts=prompts,
            trainer=trainer,
            inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=512),
            text_only=True,
        )
    ```

    Args:
        path (Union[Path, str]): The path to the model checkpoint.
        prompts (list[str]): The list of prompts to generate text for.
        trainer (nl.Trainer): The trainer object.
        encoder_prompts (Optional[list[str]], optional): The list of encoder prompts. Defaults to None.
        params_dtype (torch.dtype, optional): The data type of the model parameters. Defaults to torch.bfloat16.
        add_BOS (bool, optional): Whether to add the beginning of sequence token. Defaults to False.
        max_batch_size (int, optional): The maximum batch size. Defaults to 4.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        inference_batch_times_seqlen_threshold (int, optional): If batch-size times sequence-length is smaller than
            this threshold then we will not use pipelining, otherwise we will. Defaults to 1000.
        inference_params (Optional["CommonInferenceParams"], optional): The inference parameters defined in
            Mcore's CommonInferenceParams. Defaults to None.
        text_only (bool, optional): Whether to return only the generated text as a string. Defaults to False.

    Returns:
        list[Union["InferenceRequest", str]]: A list of generated text,
            either as a string or as an InferenceRequest object.
    """
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
