"""Base model classes for CVNN framework."""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """
    Base class for all CVNN models.

    Provides common functionality for model configuration, parameter counting,
    and size information. All CVNN models should inherit from this class.

    Args:
        **kwargs: Model configuration parameters
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, x: Tensor) -> Any:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Copy of model configuration dictionary
        """
        return self.config.copy()

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> Dict[str, int]:
        """
        Get comprehensive model size information.

        Returns:
            Dictionary with total, trainable, and non-trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
        }


class BaseEncoder(BaseModel):
    """Base class for encoder models."""

    @abstractmethod
    def encode(self, x: Tensor) -> Any:
        """Encode input to latent representation.

        Return type is intentionally flexible (Any) because encoders may
        return tensors or tuples (e.g., tensor + aux info) depending on
        concrete implementations.
        """
        pass


class BaseDecoder(BaseModel):
    """Base class for decoder models."""

    @abstractmethod
    def decode(self, z: Tensor) -> Any:
        """Decode latent representation to output.

        Return type is intentionally flexible (Any) because decoders may
        return tensors or tuples (e.g., reconstructed tensor + projection).
        """
        pass


class BaseAutoEncoder(BaseModel):
    """
    Base class for autoencoder models.

    Provides standard encoder-decoder architecture with encode/decode methods.
    Subclasses should implement the abstract encode and decode methods.

    Args:
        **kwargs: Model configuration parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder: Optional[BaseEncoder] = None
        self.decoder: Optional[BaseDecoder] = None

    @abstractmethod
    def encode(self, x: Tensor) -> Any:
        """Encode input to latent representation.

        Flexible return to allow encoders that produce auxiliary outputs
        such as downsampling probabilities or skip-connections.
        """
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Any:
        """Decode latent representation to output.

        Flexible return to allow decoders that may return (output, projected)
        tuples for convenience.
        """
        pass

    def forward(self, x: Tensor) -> Any:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        return self.decode(z)

    def get_latent_representation(self, x: Tensor) -> Any:
        """Get latent representation without decoding."""
        return self.encode(x)


class BaseVariationalAutoEncoder(BaseAutoEncoder):
    """Base class for variational autoencoder models."""

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to mean and log variance."""
        pass

    @abstractmethod
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE."""
        pass

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning reconstruction, mu, and logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class BaseComplexModel(BaseModel):
    """
    Base class for complex-valued models.

    Supports different layer modes: 'complex', 'real', and 'split'.
    Provides utilities for working with complex-valued tensors.

    Args:
        layer_mode: Mode for layers ('complex', 'real', 'split')
        **kwargs: Additional model configuration parameters

    Raises:
        ValueError: If layer_mode is not one of the supported values
    """

    def __init__(self, layer_mode: str = "complex", **kwargs):
        super().__init__(layer_mode=layer_mode, **kwargs)
        self.layer_mode = layer_mode

        if layer_mode not in ["complex", "real", "split"]:
            raise ValueError(
                f"Invalid layer_mode: {layer_mode}. Choose from 'complex', 'real', 'split'."
            )

    def get_conv_mode(self) -> str:
        """Get the convolution mode."""
        return self.layer_mode

    @staticmethod
    def is_complex_tensor(x: Tensor) -> bool:
        """Check if tensor is complex-valued."""
        return x.dtype in [torch.complex64, torch.complex128]
