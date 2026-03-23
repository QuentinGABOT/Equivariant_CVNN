# Standard library imports
from typing import Optional, Any, Union, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn
from torch import Tensor
import math
from sklearn.mixture import BayesianGaussianMixture
import torch.distributions as D
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA   

# Local imports
from .conv import DoubleConv, SingleConv
from .linear import DoubleLinear, SingleLinear
from .utils import (
    get_downsampling,
    get_dropout,
    get_upsampling,
    get_projection,
    is_real_mode,
)
from .learn_poly_sampling.layers import PolyphaseInvariantUp2D, PolyphaseInvariantDown2D
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


class Down(nn.Module):
    """
    Downscaling block for U-Net architecture.

    Applies downsampling (pooling or strided convolution) followed by
    multiple convolution blocks with optional dropout.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)
        downsampling: Downsampling method ('maxpool', 'avgpool', etc.)
        downsampling_factor: Factor for spatial dimension reduction
        residual: Whether to use residual connections
        dropout: Dropout probability (0.0 to disable)
        num_blocks: Number of successive convolution blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        layer_mode: str,
        num_blocks: int,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        normalization: str = None,
        downsampling: str = None,
        downsampling_factor: int = 2,
        kernel_size: int = 3,
        stride: Union[int, str] = 1,
        residual: bool = False,
        dropout: float = 0.0,
        gumbel_softmax: str = None,
    ) -> None:
        """Initialize downscaling block."""
        super().__init__()

        # --- LPD / Downsampling Logic ---
        if downsampling in ["LPD", "LPD_F"]:
            lpd_conv = DoubleConv(
                in_ch=out_channels,
                out_ch=out_channels,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=residual,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            lpd_conv = None

        self.down = get_downsampling(
            downsampling=downsampling or None,
            projection=projection or None,
            projection_config=projection_config,
            factor=downsampling_factor,
            layer_mode=layer_mode,
            conv=lpd_conv,
            in_channels=out_channels,
            out_channels=out_channels,
            gumbel_softmax_type=gumbel_softmax,
        )

        # --- Dynamic Convolution Blocks ---
        # We perform `num_blocks` convolutions.
        # The first (num_blocks - 1) maintain in_channels.
        # The last one projects to out_channels.
        layers = []
        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)
            # --- Stride/Padding Logic ---
            if is_last and downsampling is None:
                stride = downsampling_factor
                
            layers.append(
                DoubleConv(
                    in_ch=in_channels,
                    out_ch=out_channels if is_last else in_channels,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=residual,
                    stride=stride,
                    padding=kernel_size // 2,
                    kernel_size=kernel_size,
                )
            )
        
        self.convs = nn.Sequential(*layers)
        self.dropout = get_dropout(dropout, layer_mode, spatial=True)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        # Apply sequential blocks
        x = self.convs(x)
        if isinstance(self.down, PolyphaseInvariantDown2D):
            x, prob = self.down(x, ret_prob=True)
        else:
            prob = None
            x = self.down(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, prob

class Up(nn.Module):
    """
    Upscaling block for U-Net architecture.

    Applies upsampling (transpose convolution or interpolation) followed by
    multiple convolution blocks. Supports skip connections.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        upsampling: Upsampling method ('transpose', 'interpolate', etc.)
        skip_connections: Whether to concatenate skip connections
        normalization: Type of normalization ('batch', 'instance', etc.)
        upsampling_factor: Factor for spatial dimension increase
        residual: Whether to use residual connections
        num_blocks: Number of successive convolution blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        layer_mode: str,
        num_blocks: int,
        upsampling: Optional[str],
        skip_connection: bool = False,
        normalization: str = None,
        upsampling_factor: int = 2,
        kernel_size: int = 3,
        residual: bool = False,
        gumbel_softmax: Optional[str] = None,
    ) -> None:
        """Initialize upscaling block."""
        super().__init__()

        # Handle channel count for conv layer input
        if upsampling != "transpose":
            self.conv_adjust = DoubleConv(
                in_ch=in_channels,
                out_ch=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=residual,
            )
            self.up = get_upsampling(
                upsampling=upsampling,
                factor=upsampling_factor,
                layer_mode=layer_mode,
                in_channels=out_channels,
                out_channels=out_channels,
                gumbel_softmax_type=gumbel_softmax,
            )
        else:
            self.up = get_upsampling(
                upsampling=upsampling,
                factor=upsampling_factor,
                layer_mode=layer_mode,
                in_channels=in_channels,
                out_channels=out_channels,
                gumbel_softmax_type=gumbel_softmax,
            )
            self.conv_adjust = None
        
        in_channels = out_channels # Regardless of upsampling, after up we have out_channels as transpose conv or conv_adjust adjust the channels

        if skip_connection:
            in_channels += out_channels

        # --- Dynamic Convolution Blocks ---
        # The first (num_blocks - 1) maintain the effective in_channels (including skip).
        # The last one projects to out_channels.
        layers = []
        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)

            layers.append(
                DoubleConv(
                    in_ch=in_channels,
                    out_ch=out_channels if is_last else in_channels,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=residual,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        
        self.convs = nn.Sequential(*layers)

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, prob: Optional[Tensor] = None) -> Tensor:
        """Apply upsampling and convolution."""
            
        if isinstance(self.up, c_nn.ConvTranspose2d) or isinstance(self.up, nn.ConvTranspose2d):
            x1 = self.up(x1)
        else:
            x1 = self.conv_adjust(x1)            
            if isinstance(self.up, PolyphaseInvariantUp2D):
                x1 = self.up(x1, prob=prob)
            else:
                x1 = self.up(x1)            
                
        x = concat(x1, x2)
        x = self.convs(x)
        return x
    
class FullyConnected(nn.Module):
    """
    Fully connected layer for classifier architectures.

    Compresses spatial feature maps to a lower-dimensional latent space
    representation. Supports both real and complex-valued data.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimensionality of latent space
        input_size: Spatial size of input (assumes square)
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)

    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation: nn.Module,
        layer_mode: str = "complex",
        normalization: Optional[str] = None,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        dropout: Optional[float] = None,
    ) -> None:

        """Initialize latent bottleneck layer."""
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.normalization = normalization
        self.dropout = get_dropout(dropout, layer_mode, spatial=False)

        # store mode/activation for potential lazy re-init of fc_1
        self._activation = activation
        self._normalization = normalization
        self._layer_mode = layer_mode

        # Initial fc_1 expects flattened pooled features of size `in_channels` -> latent_dim
        self.fc_1 = DoubleLinear(
            in_ch=in_channels,
            mid_ch=in_channels,
            out_ch=num_classes,
            linear_mode=layer_mode,
            activation=activation,
            normalization=normalization,
        )

        # Use real AvgPool2d for real mode and complex AvgPool2d for complex mode
        if is_real_mode(layer_mode):
            # For real tensors use PyTorch AvgPool2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            # For complex tensors use torchcvnn AvgPool2d
            self.avg_pool = c_nn.AdaptiveAvgPool2d(1)
        self.projection = get_projection(
            projection=projection,
            layer_mode=layer_mode,
            projection_config=projection_config,
        )

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        Forward pass through the bottleneck.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Prediction (B, num_classes)
        """
        if len(x.shape) == 4:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_1(x)
        x_projected = self.projection(x)
        return x, x_projected

def concat(x1, x2):
    """
    Concatenate two tensors with automatic padding for size matching.

    Pads x1 to match x2's spatial dimensions, then concatenates along
    the channel dimension. Used in U-Net style architectures.

    Args:
        x1: First tensor (CHW format)
        x2: Second tensor (CHW format) or None

    Returns:
        Concatenated tensor along channel dimension, or x1 if x2 is None
    """
    if x2 is None:
        return x1
    else:
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
