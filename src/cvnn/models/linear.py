# Standard library imports
from typing import Optional

# Third-party imports
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn
from torch import Tensor

# Local imports
from .utils import get_activation, get_normalization

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float64


class BaseLinear(nn.Module):
    """
    Base linear layer supporting complex and real modes.

    Provides a unified interface for linear layers in different modes:
    - 'complex': Uses complex-valued linear operations
    - 'real' or 'split': Uses real-valued linear operations

    Args:
        in_ch: Number of input features
        out_ch: Number of output features
        bias: Whether to use bias term
        linear_mode: Linear layer mode ('complex', 'real', 'split')
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias: bool = True,
        linear_mode: str = "complex",
    ) -> None:
        super().__init__()
        self.linear_mode = linear_mode
        if self.linear_mode == "complex":
            self.linear = nn.Linear(
                in_features=in_ch,
                out_features=out_ch,
                bias=bias,
                dtype=COMPLEX_DTYPE,
            )
        elif self.linear_mode in ["real", "split"]:
            self.linear = nn.Linear(
                in_features=in_ch,
                out_features=out_ch,
                bias=bias,
            )
        else:
            raise ValueError(
                f"Unknown linear_mode: {self.linear_mode}. Choose from 'complex', 'real', or 'split'."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transformation to complex input, returning appropriate output."""
        if self.linear_mode == "complex":
            return self.linear(x)
        elif self.linear_mode == "real":
            return self.linear(x)
        elif self.linear_mode == "split":
            # split real and imaginary parts
            x_real = x.real
            x_imag = x.imag
            # apply linear transformation to both parts
            out_real = self.linear(x_real)
            out_imag = self.linear(x_imag)
            # return complex output
            return torch.complex(out_real, out_imag)
        else:
            raise ValueError(f"Unknown linear_mode: {self.linear_mode}")


class LinearBlock(nn.Module):
    """Linear block for complex-valued data."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias: bool = True,
        linear_mode: str = "complex",
        activation: nn.Module = None,
        normalization: str = None,
    ) -> None:
        """Initialize linear block."""
        super().__init__()
        layers = [
            BaseLinear(in_ch=in_ch, out_ch=out_ch, bias=bias, linear_mode=linear_mode)
        ]

        normalization_layer = get_normalization(
            norm_type=normalization, num_features=out_ch, layer_mode=linear_mode, dimensionality=1
        )
        layers += [
            normalization_layer,
        ]
        activation_layer = get_activation(activation, linear_mode)

        layers += [
            activation_layer,
        ]
        self.linear_block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through linear block."""
        return self.linear_block(x)


class SingleLinear(nn.Module):
    """Single linear layer for complex-valued data."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias: bool = True,
        linear_mode: str = "complex",
        activation: nn.Module = None,
        normalization: str = None,
    ) -> None:
        """Initialize single linear layer."""
        super().__init__()
        self.linear = LinearBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            bias=bias,
            linear_mode=linear_mode,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through single linear layer."""
        return self.linear(x)


class DoubleLinear(nn.Module):
    """Double linear layer for complex-valued data."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: int = None,
        bias: bool = True,
        linear_mode: str = "complex",
        activation: nn.Module = None,
        normalization: str = None,
    ) -> None:
        """Initialize double linear layer."""
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.block = nn.Sequential(
            LinearBlock(
                in_ch=in_ch,
                out_ch=mid_ch,
                bias=bias,
                linear_mode=linear_mode,
                activation=activation,
                normalization=normalization,
            ),
            LinearBlock(
                in_ch=mid_ch,
                out_ch=out_ch,
                bias=bias,
                linear_mode=linear_mode,
                activation=activation,
                normalization=normalization,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through double linear layer."""
        return self.block(x)
