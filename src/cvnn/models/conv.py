# Standard library imports
from typing import Optional, Any, Union, Tuple, cast

# Third-party imports
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn
from torch import Tensor

# Local imports
from .utils import get_normalization, get_projection, get_activation

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float64


class BaseConv2d(nn.Module):
    """
    Base 2D convolution layer supporting complex and real modes.

    Provides a unified interface for convolutions in different modes:
    - 'complex': Uses complex-valued convolutions
    - 'real' or 'split': Uses real-valued convolutions

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride for convolution
        padding: Padding for convolution
        bias: Whether to use bias term
        padding_mode: Padding mode ('replicate', 'zeros', etc.)
        conv_mode: Convolution mode ('complex', 'real', 'split')
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: Union[int, str] = 1,
        padding: Union[int, str] = 0,
        bias: bool = True,
        padding_mode: str = "replicate",
        conv_mode: str = "complex",
    ) -> None:
        super().__init__()
        self.conv_mode = conv_mode
        if self.conv_mode == "complex":
            self.conv: nn.Module = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=cast(int, stride),
                padding=cast(int, padding),
                bias=bias,
                padding_mode=padding_mode,
                dtype=COMPLEX_DTYPE,
            )
        elif self.conv_mode in ["real", "split"]:
            self.conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=cast(int, stride),
                padding=cast(int, padding),
                bias=bias,
                padding_mode=padding_mode,
            )
        else:
            raise ValueError(
                f"Unknown conv_mode: {self.conv_mode}. Choose from 'complex', 'real', or 'split'."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution to complex input, returning complex output."""
        if self.conv_mode == "complex":
            return self.conv(x)
        elif self.conv_mode == "real":
            return self.conv(x)
        elif self.conv_mode == "split":
            # split real and imaginary parts
            x_real = x.real
            x_imag = x.imag
            # apply convolution to both parts
            out_real = self.conv(x_real)
            out_imag = self.conv(x_imag)
            # return complex output
            return torch.complex(out_real, out_imag)
        else:
            raise ValueError(f"Unknown conv_mode: {self.conv_mode}")


class ConvBlock(nn.Module):
    """
    Convolution block with normalization and activation.

    Combines convolution, normalization, and activation layers in sequence.
    Supports both complex and real-valued operations.

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        normalization: Normalization layer or None
        activation: Activation function module
        kernel_size: Size of convolution kernel
        stride: Stride for convolution
        padding: Padding for convolution
        padding_mode: Padding mode for convolution
        bias: Whether to use bias in convolution
        conv_mode: Convolution mode ('complex', 'real', 'split')
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        normalization: str,
        activation: str = None,
        kernel_size: int = 3,
        stride: Union[int, str] = 1,
        padding: Union[int, str] =1,
        padding_mode: str = "circular",
        bias: bool = True,
        conv_mode: str = "complex",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            BaseConv2d(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
                conv_mode=conv_mode,
            ),
        ]

        normalization_layer = get_normalization(
            norm_type=normalization, num_features=out_ch, layer_mode=conv_mode
        )
        layers += [
            normalization_layer,
        ]
        activation_layer = get_activation(activation, conv_mode)
        layers += [
            activation_layer,
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class SingleConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_mode: str,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        activation: str = None,
        normalization: str = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            conv_mode=conv_mode,
            normalization=normalization,
        )
        self.projection = get_projection(
            projection=projection,
            layer_mode=conv_mode,
            projection_config=projection_config,
        )

    def forward(self, x) -> Union[Tensor, Tuple[Tensor, Any]]:
        x = self.conv(x)
        if isinstance(self.projection, nn.Identity):
            # If projection is identity, just return the input
            return x
        # Otherwise, apply projection
        else:
            return x, self.projection(x)

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        conv_mode: str,
        activation: str = None,
        normalization: str = None,
        mid_ch: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = 1,
        bias: bool = True,
        residual: bool = False,
        padding_mode: str = "circular",
    ) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.residual = residual

        self.block = nn.Sequential(
            ConvBlock(
                normalization=normalization,
                in_ch=in_ch,
                out_ch=mid_ch,
                stride=1,
                activation=activation,
                conv_mode=conv_mode,
                padding=padding,
                kernel_size=kernel_size,
                bias=bias,
                padding_mode=padding_mode,
            ),
            ConvBlock(
                normalization=normalization,
                in_ch=mid_ch,
                out_ch=out_ch,
                stride=stride,
                activation=activation,
                conv_mode=conv_mode,
                padding=padding,
                kernel_size=kernel_size,
                bias=bias,
                padding_mode=padding_mode,
            ),
        )

        # Connection for residual blocks
        if self.residual:
            if in_ch != out_ch or stride != 1:
                kernel_size = 1
                # Use 1x1 convolution to match dimensions
                self.residual_connection: nn.Module = BaseConv2d(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    conv_mode=conv_mode,
                    padding_mode=padding_mode,
                )
            else:
                # Direct skip connection when dimensions match
                self.residual_connection = nn.Identity()

    def forward(self, x) -> Tensor:
        x_out = self.block(x)

        if self.residual:
            return x_out + self.residual_connection(x)
        else:
            return x_out