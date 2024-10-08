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
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseVAE(torch.nn.Module, ABC):
    """
    Abstract base class for a Variational Autoencoder (VAE).

    All subclasses should implement the methods to define the behavior for encoding
    and decoding, along with specifying the latent channel size.
    """

    def __init__(self, channel: int = 3, name: str = "vae"):
        super().__init__()
        self.channel = channel
        self.name = name

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the VAE.
        """
        return self.channel

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a latent representation.

        Args:
        - state (torch.Tensor): The input tensor to encode.

        Returns:
        - torch.Tensor: The encoded latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the original space.

        Args:
        - latent (torch.Tensor): The latent tensor to decode.

        Returns:
        - torch.Tensor: The decoded tensor.
        """
        pass

    @property
    def spatial_reduce(self) -> int:
        """
        Returns the spatial reduction factor for the VAE.
        """
        # TODO: (qsh 2024-07-23) This should be a property of the VAE, not the base class.
        # Since all our VAEs have a spatial reduction factor of 16, we can hardcode it here.
        return 16


class JITVAE(BaseVAE):
    """
    A JIT compiled Variational Autoencoder (VAE) that loads pre-trained encoder
    and decoder components from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The JIT compiled encoder loaded from storage.
        decoder (Module): The JIT compiled decoder loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        name (str): Name of the model, used for differentiating cache file paths.
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(latent_ch, name)
        dtype = torch.bfloat16 if is_bf16 else torch.float32
        self.dtype = dtype
        self.is_image = is_image
        self.mean_std_fp = mean_std_fp
        self.name = name

        self.register_mean_std(mean_std_fp)
        self.load_encoder(enc_fp)
        self.load_decoder(dec_fp)
    

    def load_encoder(self, enc_fp: str) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - enc_fp (str): File path to the encoder's JIT file on the remote store.
        """
        self.encoder = self.load_jit_model(enc_fp).to(self.dtype).cuda()
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

    def load_decoder(self, dec_fp: str) -> None:
        """
        Load the decoder from the remote store.

        Args:
        - dec_fp (str): File path to the decoder's JIT file on the remote store.
        """
        self.decoder = self.load_jit_model(dec_fp).to(self.dtype).cuda()
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)
    
    def load_jit_model(self, file_path):
        """
        Loads a JIT model from a specified file path.

        Args:
        - file_path (str): Path to the JIT model file.

        Returns:
        - torch.jit.ScriptModule: The loaded JIT model.
        """
        try:
            model = torch.jit.load(file_path, map_location='cpu')
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None
    
    def load_mean_std(self, mean_std_fp: str):
        mean, std = torch.load(mean_std_fp, map_location='cpu')
        return mean, std

    def register_mean_std(self, mean_std_fp: str) -> None:
        latent_mean, latent_std = self.load_mean_std(mean_std_fp)
        latent_mean, latent_std = latent_mean.to(self.dtype).cuda(), latent_std.to(self.dtype).cuda()
        target_shape = [1, self.latent_ch, 1, 1] if self.is_image else [1, self.latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            latent_mean.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the input state to latent space; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = state.dtype
        latent_mean = self.latent_mean.to(in_dtype)
        latent_std = self.latent_std.to(in_dtype)
        encoded_state = self.encoder(state.to(self.dtype))
        if isinstance(encoded_state, torch.Tensor):
            pass
        elif isinstance(encoded_state, tuple):
            assert isinstance(encoded_state[0], torch.Tensor)
            encoded_state = encoded_state[0]
        else:
            raise ValueError("Invalid type of encoded state")
        return (encoded_state.to(in_dtype) - latent_mean) / latent_std

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode the input latent to state; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = latent.dtype
        latent = latent * self.latent_std.to(in_dtype) + self.latent_mean.to(in_dtype)
        return self.decoder(latent.to(self.dtype)).to(in_dtype)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.encoder.to(self.dtype)