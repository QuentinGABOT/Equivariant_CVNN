# Standard library imports
from typing import List, Optional, Any, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint_sequential

# Local imports
from .base import BaseEncoder, BaseAutoEncoder, BaseComplexModel
from .blocks import Down, LatentBottleneck, Up, FullyConnected, VariationalBottleneck
from .conv import SingleConv
from .utils import (
    init_weights_mode_aware,
)

# COMPLEX_DTYPE: torch.dtype = torch.complex64
# REAL_DTYPE: torch.dtype = torch.float32
DOWNSAMPLING_FACTOR = 2
UPSAMPLING_FACTOR = 2

__all__ = [
    "AutoEncoder",
    "LatentAutoEncoder",
    "VariationalAutoEncoder",
    "UNet",
    "ResNet",
]


class AutoEncoder(BaseAutoEncoder, BaseComplexModel):
    """Autoencoder with downsampling and upsampling layers using complex convolutions."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        upsampling_layer: str,
        num_blocks: int,
        layer_mode: str = "complex",
        normalization_layer: Optional[str] = None,
        downsampling_layer: Optional[str] = None,
        residual: bool = False,
        dropout: float = 0.0,
        projection_layer: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the autoencoder."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
            **kwargs,
        )
        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connection=False,
            upsampling=True,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )
        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor, return_probs: bool = True) -> Tensor:
        """Encode input to latent representation."""
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
                list_probs.append(prob)
            else:
                x = enc(x)
        if return_probs:
            return x, list_probs
        else:
            return x

    def decode(self, z: Tensor, probs: Optional[list] = None) -> Tensor:
        """Decode latent representation to output."""
        probs_iter = iter(reversed(probs)) if probs is not None else None
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                z = dec(z, prob=next(probs_iter) if probs_iter is not None else None)
            else:
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder and decoder."""
        x, list_probs = self.encode(x, return_probs=True)
        x = self.decode(x, probs=list_probs)
        return x

class LatentAutoEncoder(BaseAutoEncoder, BaseComplexModel):
    """Autoencoder with a latent vector bottleneck between encoder and decoder."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        latent_dim: int,
        num_blocks: int,
        upsampling_layer: str,
        layer_mode: str = "complex",
        normalization_layer: str = None,
        downsampling_layer: str = None,
        residual: bool = False,
        dropout: float = 0.0,
        projection_layer: str = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the latent autoencoder."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            latent_dim=latent_dim,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
            **kwargs,
        )

        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connection=False,
            upsampling=True,
            bottleneck=True,
            latent_dim=latent_dim,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )

        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor, return_probs: bool = False) -> Tensor:
        """Encode input to latent representation."""
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
                list_probs.append(prob)
            else:
                x = enc(x)
        if return_probs:
            return x, list_probs
        else:
            return x

    def bottleneck(self, x: Tensor) -> Tensor:
        """Get latent representation from bottleneck."""
        return self.convnet.bottleneck(x)

    def decode(self, z: Tensor, probs: Optional[list] = None) -> Tensor:
        """Decode latent representation to output."""
        probs_iter = iter(reversed(probs)) if probs is not None else None
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                z = dec(z, prob=next(probs_iter) if probs_iter is not None else None)
            else:
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder, bottleneck, and decoder."""
        x, list_probs = self.encode(x, return_probs=True)
        x = self.bottleneck(x)
        x = self.decode(x, probs=list_probs)
        return x

class VariationalAutoEncoder(BaseAutoEncoder, BaseComplexModel):
    """VariationalAutoEncoder."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        latent_dim: int,
        num_blocks: int,
        upsampling_layer: Optional[str] = None,
        layer_mode: str = "complex",
        cov_mode: str = "diag",
        normalization_layer: Optional[str] = None,
        downsampling_layer: Optional[str] = None,
        residual: bool = False,
        dropout: float = 0.0,
        projection_layer: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        force_circular: bool = False,
        standard_reparam: bool = False,
        use_conv_1x1: bool = False,
        learned_variance: bool = False,
        sample_gmm: bool = False,
    ) -> None:
        """Initialize the variational autoencoder."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            latent_dim=latent_dim,
            layer_mode=layer_mode,
            cov_mode=cov_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
            force_circular=force_circular,
            standard_reparam=standard_reparam,
            use_conv_1x1=use_conv_1x1,
        )

        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            cov_mode=cov_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connection=False,
            upsampling=True,
            bottleneck=True,
            variational=True,
            latent_dim=latent_dim,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
            force_circular=force_circular,
            standard_reparam=standard_reparam,
            use_conv_1x1=use_conv_1x1,
        )
        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))
        self.sample_gmm = sample_gmm
        if learned_variance:
            self.log_sigma2_dec = nn.Parameter(torch.zeros(1))
        else:
            self.log_sigma2_dec = torch.tensor(0.0)

    def encode(self, x: Tensor, return_probs: bool = False) -> Union[Tensor, Tuple[Tensor, list]]:
        """Encode input to latent representation.

        Args:
            x: input tensor
            return_probs: if True return (x, list_probs) where list_probs is
                the per-layer downsampling probs produced by `Down` blocks.
        """
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
                list_probs.append(prob)
            else:
                x = enc(x)
        if return_probs:
            return x, list_probs
        return x

    def bottleneck(self, x: Tensor) -> Tensor:
        """Get latent representation from bottleneck."""
        return self.convnet.bottleneck(x)

    def decode(self, z: Tensor, probs: Optional[list] = None) -> Tensor:
        """Decode latent representation to output.

        If `probs` is provided it should be the list of probabilities produced
        by `encode(..., return_probs=True)` and will be fed (reversed) to the
        upsampling layers.
        """
        probs_iter = iter(reversed(probs)) if probs is not None else None
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                z = dec(z, prob=next(probs_iter) if probs_iter is not None else None)
            else:
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder, bottleneck, and decoder."""
        x, list_probs = self.encode(x, return_probs=True)
        x, z, mu, p1, p2 = self.bottleneck(x)
        return self.decode(x, probs=list_probs), mu, p1, p2, self.log_sigma2_dec

    def sample(self, num_samples: int) -> Tensor:
        """Sample from the latent space and decode to generate new data.
        
        Args:
            num_samples: Number of images to generate
        """
        # Sample z
        z = self.convnet.bottleneck[0].sample(num_samples, self.sample_gmm)
        bottleneck_module = self.convnet.bottleneck[0]
        z = bottleneck_module.to_input(z)
        x_recon = self.decode(z)
        return x_recon
        
class UNet(BaseAutoEncoder, BaseComplexModel):
    """UNet model with downsampling and upsampling layers using complex convolutions."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        num_classes: int,
        num_blocks: int,
        layer_mode: str = "complex",
        normalization_layer: str = None,
        downsampling_layer: str = "maxpool",
        upsampling_layer: str = "nearest",
        residual: bool = False,
        projection_layer: str = "amplitude",
        projection_config: Optional[dict] = None,
        dropout: float = 0.0,
        gumbel_softmax: Optional[str] = None,
    ) -> None:
        """Initialize the UNet model."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            num_classes=num_classes,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )

        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            num_classes=num_classes,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connection=True,
            upsampling=True,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )
        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor, return_probs: bool = False) -> Union[Tensor, Tuple[Tensor, list, list]]:
        """Encode input to latent representation.

        For UNet we also collect skip connections and per-layer downsampling
        probabilities produced by `Down` blocks. If `return_probs=True` this
        returns a tuple (encoded, list_probs, list_skip_connections) to be
        used by the decoder.
        """
        list_skip_connections: list[Any] = []
        list_probs: list[Any] = []

        # initialize with None to match decoder alignment used elsewhere
        list_skip_connections.append(None)
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
                list_probs.append(prob)
            else:
                x = enc(x)
            list_skip_connections.append(x)

        # Pass through bridge and include its probability
        x, prob = self.convnet.bridge(x)
        list_probs.append(prob)

        if return_probs:
            return x, list_probs, list_skip_connections
        else:
            return x, list_skip_connections

    def decode(
        self, z: Tensor, probs: Optional[list] = None, skips: Optional[list] = None
    ) -> Tuple[Tensor, Any]:
        """Internal UNet decoder implementation used by forward/decode.

        Args:
            z: tensor input to decoder
            probs: list of per-layer probs from encoder (optional)
            skips: list of skip-connection tensors from encoder (optional)

        Returns:
            (x, x_projected)
        """
        probs_iter = iter(reversed(probs)) if probs is not None else None
        skips_iter = iter(reversed(skips)) if skips is not None else None
        x_projected = None
        x = z
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                prob = next(probs_iter) if probs_iter is not None else None
                skip = next(skips_iter) if skips_iter is not None else None
                x = dec(x1=x, x2=skip, prob=prob)
            else:
                out = dec(x)
                if isinstance(out, tuple) and len(out) == 2:
                    x, x_projected = out
                else:
                    x = out
        return x, x_projected

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Forward pass through encoder and decoder."""
        # Use the encode helper to collect skip connections and probs
        x, list_probs, list_skip_connections = self.encode(x, return_probs=True)
        x, x_projected = self.decode(x, probs=list_probs, skips=list_skip_connections)
        # UNet always returns (x, x_projected)
        return x, x_projected

class ResNet(BaseEncoder, BaseComplexModel):
    def __init__(self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        num_classes: int,
        num_blocks: int,
        layer_mode: str = "complex",
        normalization_layer: str = None,
        downsampling_layer: str = "maxpool",
        residual: bool = False,
        projection_layer: str = "amplitude",
        projection_config: Optional[dict] = None,
        dropout: float = 0.0,
        gumbel_softmax: Optional[str] = None) -> None:
        """Initialize the ResNet model."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            residual=residual,
            num_classes=num_classes,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )
        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            num_classes=num_classes,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=None,
            residual=residual,
            skip_connection=False,
            upsampling=False,
            bottleneck=True,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
            num_blocks=num_blocks,
        )
        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, _ = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
        return x

    def bottleneck(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Get latent representation from bottleneck."""
        return self.convnet.bottleneck(x)

    def forward(self, x):
        """Forward pass through encoder and bottleneck."""
        # Use encode helper to collect probabilities, then call bottleneck
        x = self.encode(x)
        # convnet.bottleneck may return (tensor, projected) — return it as-is
        return self.convnet.bottleneck(x)

class ConvNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        num_blocks: int,
        num_classes: Optional[int] = None,
        projection_layer: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        normalization_layer: Optional[str] = None,
        downsampling_layer: Optional[str] = None,
        upsampling_layer: Optional[str] = None,
        dropout: float = 0.0,
        latent_dim: int = 0,
        residual: bool = False,
        skip_connection: bool = False,
        upsampling: bool = False,
        bottleneck: bool = False,
        variational: bool = False,
        layer_mode: str = "complex",
        cov_mode: str = None,
        force_circular : bool = False,
        standard_reparam: bool = False,
        use_conv_1x1:  bool = False,
    ) -> None:
        """ConvNet with downsampling and upsampling layers using complex convolutions."""
        super().__init__()
        assert bottleneck or upsampling, "Either dense or upsampling must be provided"

        # Encoder/decoder layer builders
        current_channels = channels_width
        encoder_layers: List[Any] = []
        bridge_layers: List[Any] = []
        bottleneck_layers: List[Any] = []
        decoder_layers: List[Any] = []

        encoder_layers.append(
            SingleConv(
                in_ch=num_channels, out_ch=current_channels, conv_mode=layer_mode, projection=None
            )
        )
        out_channels = current_channels

        for i in range(1, num_layers + 1):
            out_channels *= 2
            if i < num_layers:
                encoder_layers.append(
                    Down(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        layer_mode=layer_mode,
                        normalization=normalization_layer,
                        downsampling=downsampling_layer,
                        downsampling_factor=DOWNSAMPLING_FACTOR,
                        residual=residual,
                        dropout=dropout,
                        projection=projection_layer,
                        projection_config=projection_config,
                        gumbel_softmax=gumbel_softmax,
                        num_blocks=num_blocks,
                    )
                )
            else:
                if skip_connection:
                    bridge_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                            downsampling=downsampling_layer,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                            residual=residual,
                            dropout=dropout,
                            projection=projection_layer,
                            projection_config=projection_config,
                            gumbel_softmax=gumbel_softmax,
                            num_blocks=num_blocks,
                        )
                    )
                else:
                    encoder_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                            downsampling=downsampling_layer,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                            residual=residual,
                            dropout=dropout,
                            projection=projection_layer,
                            projection_config=projection_config,
                            gumbel_softmax=gumbel_softmax,
                            num_blocks=num_blocks,
                        )
                    )
            current_channels = out_channels

        # Build nn.Sequential modules
        self.encoder: Any = nn.Sequential(*encoder_layers)
        self.bridge: Any = nn.Sequential(*bridge_layers)

        if not upsampling:
            bottleneck_input_size = input_size // (DOWNSAMPLING_FACTOR ** num_layers)
            bottleneck_layers.append(FullyConnected(
                in_channels=current_channels,
                num_classes=num_classes,
                input_size=bottleneck_input_size,
                activation=activation,
                layer_mode=layer_mode,
                normalization=normalization_layer,
                projection="amplitude" if projection_layer is None else projection_layer,
                projection_config=projection_config,
            ))

        else:
            if bottleneck:
                bottleneck_input_size = input_size // (DOWNSAMPLING_FACTOR ** num_layers)
                if variational:
                    bottleneck_layers.append(VariationalBottleneck(
                            in_channels=current_channels,
                            input_size=bottleneck_input_size,
                            activation=activation,
                            latent_dim=latent_dim,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                            projection=projection_layer,
                            cov_mode=cov_mode,
                            force_circular=force_circular,
                            standard_reparam=standard_reparam,
                            use_conv_1x1=use_conv_1x1,
                        ))
                    
                else:
                    bottleneck_layers.append(LatentBottleneck(
                            in_channels=current_channels,
                            input_size=bottleneck_input_size,
                            activation=activation,
                            latent_dim=latent_dim,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                        ))
                    
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        if upsampling:
            # Decoder with halving channels
            for i in range(num_layers - 1, -1, -1):
                out_channels //= 2
                decoder_layers.append(
                    Up(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        layer_mode=layer_mode,
                        normalization=normalization_layer,
                        upsampling=upsampling_layer,
                        upsampling_factor=UPSAMPLING_FACTOR,
                        skip_connection=skip_connection,
                        residual=residual,
                        gumbel_softmax=gumbel_softmax,
                        num_blocks=num_blocks,
                    )
                )
                current_channels = out_channels

            if num_classes is not None:
                # Final output layer for classification/segmentation
                decoder_layers.append(
                    SingleConv(
                        in_ch=current_channels,
                        out_ch=num_classes,
                        kernel_size=1,  # 1x1 conv for output
                        padding=0,
                        conv_mode=layer_mode,
                        projection="amplitude" if projection_layer is None else projection_layer, # we need to ensure the output is real-valued
                        projection_config=projection_config,
                    )
                )
            else:
                # Final output layer for reconstruction
                decoder_layers.append(
                    SingleConv(
                        in_ch=current_channels,
                        out_ch=num_channels,
                        kernel_size=1,  # 1x1 conv for output
                        padding=0,
                        conv_mode=layer_mode,
                        projection=None,
                        projection_config=projection_config,
                    )
                )

        self.decoder = nn.Sequential(*decoder_layers)

    def use_checkpointing(self) -> None:
        """Wrap encoder and decoder with checkpointing to save memory."""
        # wrap with CheckpointSequential to preserve Module type
        encoder_modules = list(self.encoder.children())
        decoder_modules = list(self.decoder.children())
        self.encoder = CheckpointSequential(encoder_modules)
        self.bridge = (
            CheckpointSequential(self.bridge)
            if hasattr(self, "bridge") and len(self.bridge) > 0
            else nn.Identity()
        )
        self.bottleneck = (
            CheckpointSequential(self.bottleneck)
            if hasattr(self, "bottleneck") and len(self.bottleneck) > 0
            else nn.Identity()
        )
        self.decoder = CheckpointSequential(decoder_modules)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ConvNet."""
        # Pass through encoder
        encoded = self.encoder(x)

        # Pass through bridge if it exists
        if hasattr(self, "bridge") and len(self.bridge) > 0:
            encoded = self.bridge(encoded)

        # Pass through bottleneck if it exists
        if hasattr(self, "bottleneck") and len(self.bottleneck) > 0:
            encoded = self.bottleneck(encoded)

        # Pass through decoder if it exists
        if hasattr(self, "decoder") and len(self.decoder) > 0:
            decoded = self.decoder(encoded)
            return decoded
        else:
            return encoded


class CheckpointSequential(nn.Module):
    """Wrap a sequence of modules with checkpointing."""

    def __init__(self, modules: List[nn.Module]) -> None:
        super().__init__()
        self.seq = nn.Sequential(*modules)
        self.length = len(modules)

    def forward(self, x: Tensor) -> Tensor:
        return checkpoint_sequential(self.seq, self.length, x)
