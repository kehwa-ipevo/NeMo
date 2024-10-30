import os
import time
import nemo_run as run
from nemo.collections.diffusion.data.diffusion_taskencoder import BasicDiffusionTaskEncoder
from nemo.collections.diffusion.train import multimodal_datamodule
import fiddle as fdl
import pytest
import torch
from megatron.core import parallel_state
import torch.autograd.profiler as profiler
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from nemo.collections.multimodal.data.energon.base import SimpleMultiModalDataModule
from nemo.lightning import io
from nemo.collections.llm import fn

# Fixture to initialize distributed training only once
@pytest.fixture(scope="session", autouse=True)
def initialize_distributed():
    if not torch.distributed.is_initialized():
        rank = int(os.environ['LOCAL_RANK'])
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        parallel_state.initialize_model_parallel()

# Fixture to get the value of the custom command-line option
@pytest.fixture
def path():
    return os.getenv('DATA_DIR')

def test_datamodule(path): 
    config = multimodal_datamodule()
    config.path = path
    config.num_workers = 10
    config.seq_length = 20000 # 260
    config.task_encoder.seq_length = config.seq_length
    datamodule = fdl.build(config)
    
    for i, batch in enumerate(datamodule.train_dataloader()):
        print(batch.seq_len_q)
        print(batch.video.shape)
        if i == 1:
            start_time = time.time()
        if i > 10000:
            break

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 100 batches: {elapsed_time} seconds, {elapsed_time/100} seconds per batch")


def test_datamodule2(path): 
    config = multimodal_datamodule()
    config.path = path
    config.num_workers = 10
    config.micro_batch_size = 6
    config.task_encoder.aethetic_score = 4
    config.seq_length = 8192
    config.task_encoder.seq_length = 8192
    # config.virtual_epoch_length = 0
    datamodule = fdl.build(config)

    # Define a LightningModule
    class MyModel(pl.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
        def __init__(self):
            super(MyModel, self).__init__()
            self.model = torch.nn.Linear(10, 2)  # Simple linear model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            # Do nothing
            loss = torch.tensor(0.0, requires_grad=True)
            return loss

        def configure_optimizers(self):
            # Returning a basic optimizer even though no training will happen
            return torch.optim.SGD(self.parameters(), lr=0.001)
        
    # Instantiate the model
    model = MyModel()

    # Instantiate the Trainer and fit the model
    trainer = pl.Trainer(max_epochs=5)

    trainer.fit(model, datamodule)

def test_simple_datamodule(path): 
    datamodule = SimpleMultiModalDataModule(
        path=path,
        seq_length=260,
        micro_batch_size=1,
        num_workers=16,
        tokenizer=None,
        image_processor=None,
        task_encoder=BasicDiffusionTaskEncoder(seq_length=260, text_embedding_padding_size=512,
        ),
    )

    # Define a LightningModule
    class MyModel(pl.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
        def __init__(self):
            super(MyModel, self).__init__()
            self.model = torch.nn.Linear(10, 2)  # Simple linear model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            # Do nothing
            loss = torch.tensor(0.0, requires_grad=True)
            return loss

        def configure_optimizers(self):
            # Returning a basic optimizer even though no training will happen
            return torch.optim.SGD(self.parameters(), lr=0.001)
        
    # Instantiate the model
    model = MyModel()

    # Instantiate the Trainer and fit the model
    trainer = pl.Trainer(max_epochs=5)

    trainer.fit(model, datamodule)

def test_taskencoder():
    taskencoder = BasicDiffusionTaskEncoder(
        text_embedding_padding_size=512,
        seq_length=260,
    )

    start_time = time.time()
    for i in tqdm(range(100)):
        sample = {
            'pth': torch.randn(3, 1, 30, 30),
            'pickle': np.random.randn(256, 1024),
            'json': {'image_height': 1, 'image_width': 1, 'aesthetic_score': np.random.randint(0, 10)},
        }
        taskencoder.encode_sample(sample)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 100 batches: {elapsed_time} seconds")

def test_packing(path):
    config = multimodal_datamodule()
    config.path = path
    config.num_workers = 2
    config.seq_length = 640
    config.task_encoder.seq_length = None
    config.task_encoder.max_seq_length = config.seq_length
    config.packing_buffer_size = 10
    datamodule = fdl.build(config)
    
    for i, batch in enumerate(datamodule.train_dataloader()):
        print(f'{batch.seq_len_q=}')
        print(f'{batch.video.shape=}')
        print(f'{batch.t5_text_embeddings.shape=}')
        print(f'{batch.loss_mask.shape=}')
        print(f'{batch.latent_shape.shape=}')
        # batch.seq_len_q=tensor([[221, 221]], dtype=torch.int32)
        # batch.video.shape=torch.Size([1, 640, 64])
        # batch.t5_text_embeddings.shape=torch.Size([1, 1024, 1024])
        # batch.loss_mask.shape=torch.Size([1, 640])
        # batch.latent_shape.shape=torch.Size([1, 2, 4])
        if i == 1:
            start_time = time.time()
        if i > 10000:
            break

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 100 batches: {elapsed_time} seconds, {elapsed_time/100} seconds per batch")

def test_packing2(path): 
    config = multimodal_datamodule()
    config.path = path
    config.num_workers = 8
    config.micro_batch_size = 1
    config.seq_length = 8192
    config.task_encoder.seq_length = None
    config.task_encoder.max_seq_length = config.seq_length
    config.packing_buffer_size = 1000
    config.virtual_epoch_length = 0
    # config.max_samples_per_s equence = 100

    datamodule = fdl.build(config)

    # Define a LightningModule
    seqlen = []
    class MyModel(pl.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
        def __init__(self):
            super(MyModel, self).__init__()
            self.model = torch.nn.Linear(10, 2)  # Simple linear model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            # Do nothing
            # print(batch.seq_len_q, batch.seq_len_q.sum())
            seqlen.append(batch['seq_len_q'].sum().item())
            print('avereage pack size:', sum(seqlen)/len(seqlen))
            loss = torch.tensor(0.0, requires_grad=True)
            return loss

        def configure_optimizers(self):
            # Returning a basic optimizer even though no training will happen
            return torch.optim.SGD(self.parameters(), lr=0.001)
        
    # Instantiate the model
    model = MyModel()

    # Instantiate the Trainer and fit the model
    trainer = pl.Trainer(max_epochs=5)

    trainer.fit(model, datamodule)

def test_taskencoder_packing():
    taskencoder = BasicDiffusionTaskEncoder(
        text_embedding_padding_size=512,
        seq_length=None,
        max_seq_length=8192,
    )

    start_time = time.time()
    samples = []
    for _ in tqdm(range(5)):
        for i in tqdm(range(100)):
            sample = {
                'pth': torch.randn(3, 1, 30, 30),
                'pickle': np.random.randn(256, 1024),
                'json': {'image_height': 1, 'image_width': 1, 'aesthetic_score': np.random.randint(0, 10)},
            }
            sample = taskencoder.encode_sample(sample)
            samples.append(sample)

        list_of_samples = taskencoder.select_samples_to_pack(samples)
        packed_samples = taskencoder.pack_selected_samples(list_of_samples)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 500 batches: {elapsed_time} seconds")