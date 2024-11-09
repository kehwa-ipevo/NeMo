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
    start_rest_service: bool = False,
    rest_service_http_address: str = "0.0.0.0",
    rest_service_port: int = 8000,
    openai_format_response: bool = False,
    ckpt_type: str = "nemo",
):
    from nemo.deploy import DeployPyTriton

    if start_rest_service:
        if triton_port == rest_service_port:
            logging.error("REST service port and Triton server port cannot use the same port.")
            return
        # Store triton ip, port and other args relevant for REST API in config.json to be accessible by rest_model_api.py
        store_args_to_json(triton_http_address, triton_port, triton_request_timeout, openai_format_response)

    # TODO: directly support deploy of trtllm engine wo exporting to TRTLLM
    if ckpt_type == "trtllm":
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
    elif ckpt_type == "nemo":
        if nemo_checkpoint is None:
            raise ValueError("In-Framework deployment requires a .nemo checkpoint")
        try:
            from nemo.deploy.nlp import MegatronLLMDeployable
        except Exception as e:
            raise ValueError(
                "MegatronLLMDeployable is not supported in this environment as it was not imported.{type(e).__name__}: {e}"
            )
        triton_deployable = MegatronLLMDeployable(nemo_checkpoint, num_gpus)

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
    url: str = "http://0.0.0.0:1234/v1",
    model_name: str = "xxxx",
    eval_task: str = "gsm8k",
    num_fewshot: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    # inference params
    max_tokens_to_generate: Optional[int] = 256,
    temperature: Optional[float] = None,
    top_p: Optional[float] = 0.0,
    top_k: Optional[int] = 1,
):

    import time

    import requests
    from lm_eval import evaluator

    ## This may change, how to deal with it ? In the past Instance class was in lm_eval.base
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from requests.exceptions import RequestException

    def wait_for_rest_service(rest_url, max_retries=60, retry_interval=2):
        """
        Wait for REST service to be ready.

        Args:
        rest_url (str): URL of the REST service's health endpoint
        max_retries (int): Maximum number of retry attempts
        retry_interval (int): Time to wait between retries in seconds

        Returns:
        bool: True if rest service is ready, False otherwise
        """
        for _ in range(max_retries):
            rest_ready = check_service(rest_url)

            if rest_ready:
                print("REST service is ready.")
                return True

            print(f"REST Service not ready yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

        print("Timeout: One or both services did not become ready.")
        return False

    def check_service(url):
        """
        Check if a service is ready by making a GET request to its health endpoint.

        Args:
        url (str): URL of the service's health endpoint

        Returns:
        bool: True if the service is ready, False otherwise
        """
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except RequestException:
            return False

    class CustomModel(LM):
        """
        Created based on: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.4/docs/model_guide.md
        """

        def __init__(self, model_name, api_url, max_tokens_to_generate, temperature, top_p, top_k):
            self.model_name = model_name
            self.api_url = api_url
            self.max_tokens_to_generate = max_tokens_to_generate
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            super().__init__()

        def _generate_tokens_logprobs(self, payload, return_text: bool = False, return_logprobs: bool = False):
            response = requests.post(f"{self.api_url}/completions/", json=payload)
            response_data = response.json()

            if 'error' in response_data:
                raise Exception(f"API Error: {response_data['error']}")

            # Assuming the response is in OpenAI format
            if return_text:
                return response_data['choices'][0]['text']

            if return_logprobs:
                return response_data['choices'][0]['log_probs']

        def loglikelihood(self, requests: list[Instance]):
            # log likelihood calculation logic here
            results = []
            for request in requests:
                context = request.arguments[0]
                continuation = request.arguments[1]
                full_text = context + continuation
                instance = Instance(
                    request_type="loglikelihood",
                    # doc={'text': full_text},
                    doc=request.doc,
                    arguments=(full_text,),
                    idx=0,
                )
                # Access the 'arguments' attribute of the Instance
                prompt = instance.arguments[0]  # This should be the prompt string

                # Extract default temperature from instance of the benchmark or use the user defined value
                # Does not work for MMLU since the input instance does not contain temp key
                # temperature = (
                #     instance.arguments[1].get('temperature', 1.0) if not self.temperature else self.temperature
                # )
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens_to_generate,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    # "compute_logprob": True ##TODO Do we want to have this as an
                    # user defined value or set it to True by default ?
                }

                log_probs = self._generate_tokens_logprobs(payload, return_logprobs=True)

                # Assuming log_probs is a list of log probabilities for each token
                # TODO : why is log_prbs returned as list of list ? Change it to just a list maybe in query_llm ?
                continuation_log_prob = sum(log_probs[0][0][-len(continuation) :])
                results.append((continuation_log_prob, False))

            return results

        def loglikelihood_rolling(self, requests: list[Instance]):
            # log likelihood rolling calculation logic here
            results = []
            for request in requests:
                context = request.arguments[0]
                continuation = request.arguments[1]
                full_text = context + continuation
                instance = Instance(
                    request_type="loglikelihood_rolling",
                    # doc={'text': full_text},
                    doc=request.doc,
                    arguments=(full_text,),
                    idx=0,
                )
                # Access the 'arguments' attribute of the Instance
                prompt = instance.arguments[0]  # This should be the prompt string

                # Extract default temperature from instance of the benchmark or use the user defined value
                # Does not work for MMLU since the input instance does not contain temp key
                # temperature = (
                #     instance.arguments[1].get('temperature', 1.0) if not self.temperature else self.temperature
                # )
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens_to_generate,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    # "compute_logprob": True ##TODO Do we want to have this as an
                    # user defined value or set it to True by default ?
                }

                log_probs = self._generate_tokens_logprobs(payload, return_logprobs=True)

                # Assuming log_probs is a list of log probabilities for each token
                continuation_log_probs = log_probs[0][0][-len(continuation) :]
                results.append((continuation_log_probs, False))

            return results

        def generate_until(self, inputs: list[Instance]):
            # `Instance` is a dataclass defined in [`lm_eval.api.instance`] https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.4/lm_eval/api/instance.py
            results = []
            for instance in inputs:
                # Access the 'arguments' attribute of the Instance
                prompt = instance.arguments[0]  # This should be the prompt string

                # Extract default temperature from instance of the benchmark or use the user defined value
                # Does not work for MMLU since the input instance does not contain temp key
                # temperature = (
                #     instance.arguments[1].get('temperature', 1.0) if not self.temperature else self.temperature
                # )
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens_to_generate,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    # "compute_logprob": True ##TODO Do we want to have this as an
                    # user defined value or set it to True by default ?
                }

                generated_text = self._generate_tokens_logprobs(payload, return_text=True)

                results.append(generated_text)

            return results

    wait_for_rest_service(rest_url=f"{url}/health")
    model = CustomModel(model_name, url, max_tokens_to_generate, temperature, top_p, top_k)
    results = evaluator.simple_evaluate(
        model=model, tasks=eval_task, limit=limit, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters
    )

    print("--results---", results['results'][eval_task])


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
