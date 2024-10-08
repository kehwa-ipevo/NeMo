Video Foundation Model Training Framework
=============

Overview
--------

The NeMo Video Foundation Model (VFM) Training Framework provides a scalable training platform for video diffusion models with transformer backbones.  Our new features streamline the training process, allowing developers to efficiently train state-of-the-art video models with ease. 


Some of the features we currently support include:

- Energon Dataloader for Webscale Dataloading
- Mixed Image-Video Training
- Model and Data Parallelism
- Model Architectures: DiT, MovieGen


Energon Dataloader for Webscale Dataloading
-------------------------------------------

Webscale Dataloading
^^^^^^^^^^^^^^^^^^^^

Megatron-Energon is an optimized multi-modal dataloader for large-scale deep learning with Megatron. Energon allows for distributed loading of large training training data for multi-modal model training. Energon allows for blending many datasets together and distributing the dataloading workflow across multiple cluster nodes/processes while ensuring reproducibility and resumability. It can be used across text, images, and videos with ease, providing a simple interface for VFM training.

Dataloader Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^

One of Energon's key features is its ability to save and restore its state. This functionality is crucial for long-running training processes, making the dataloader robust and recoverable after interruptions. By allowing checkpointing of the dataloader status, Energon ensures that training can be resumed from where it left off, saving time and computational resources in case of unexpected shutdowns or planned pauses in the training process. This makes it especially useful for VFM training as it requires several training jobs for end-to-end training.

Parallel Configuration
^^^^^^^^^^^^^^^^^^^^^^

Energon's architecture allows it to efficiently distribute data across multiple processing units, ensuring that each GPU or node receives a balanced workload. This parallelization not only increases the overall throughput of data processing but also helps in maintaining high utilization of available computational resources.


Mixed Image-Video Training
------------------------------

Our dataloader provides support for mixed image-video training by using the NeMo packed sequence feature to pack together images and videos of varying length into the same microbatch. The sequence packing mechanism uses the THD attention kernel, which allows us to increase the model FLOPs utilization (MFU) and efficiently process data with varying length.


(add diagram)

Model and Data Parallelism
--------------------------
NeMo VFM provides support for training models using tensor parallelism, sequence parallelism, pipeline parallelism, and context parallelism. To support pipeline parallelism with conditional diffusion training, we duplicate the conditional embeddings across the pipeline stages, and perform an all-reduce during the backward pass. This approach uses more compute, but it has a lower communication cost than sending the conditional embeddings through different pipeline stages. 


Model Architectures
-------------------

DiT
^^^
We implement an efficient version of the diffusion transformer (DiT) from (cite DiT). Our DiT is slightly modified from the original paper as we use cross attention and adaptive layernorm together in the same architecture. We also use a QK-layernorm for training stability. Our framework allows for customizing the DiT architecture while maintaining its scalability, enabling training large DiT models on long sequence lengths.


MovieGen
^^^^^^^^

