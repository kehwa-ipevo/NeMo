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

FROM nvcr.io/nvidia/pytorch:24.10-py3
COPY ../../../requirements/requirements_lightning.txt /app/requirements_lightning.txt
RUN pip install --no-cache-dir -r /app/requirements_lightning.txt
COPY ../../../requirements/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ../../../requirements/requirements_common.txt /app/requirements_common.txt
RUN pip install --no-cache-dir -r /app/requirements_common.txt
COPY ../../../requirements/requirements_nlp.txt /app/requirements_nlp.txt
RUN pip install --no-cache-dir -r /app/requirements_nlp.txt

RUN pip install --no-cache-dir git+https://github.com/NVIDIA/NeMo-Run.git
RUN pip install --no-cache-dir diffusers==0.30.3 av megatron-energon ffmpeg-python==0.2.0 imageio-ffmpeg==0.4.8 imageio==2.26.0 opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 mediapy moviepy wandb  git+https://github.com/webdataset/webdataset.git@20ef544504bcb123f9201879cd353f79d7daf173
