# Local imports

import torch
import torch.nn as nn
import torchcvnn.nn as c_nn
from .learn_poly_sampling.layers import (
    PolyphaseInvariantDown2D,
    PolyphaseInvariantUp2D,
)
from .parts import *
from .model import DeepNeuralNetwork
from .projection import *


def build_model(cfg, projection, softmax, dtype, num_classes=None, num_channels=None):
    num_layers = cfg["model"]["num_layers"]
    channels_ratio = cfg["model"]["channels_ratio"]
    latent_dim = cfg["model"]["latent_dim"]
    img_size = cfg["data"]["img_size"]
    model = cfg["model"]["class"]
    activation = cfg["model"]["activation"]
    downsampling_method = cfg["model"]["downsampling"]
    upsampling_method = cfg["model"]["upsampling"]
    normalization = cfg["model"]["normalization"]
    dropout = cfg["model"]["dropout"]
    res = cfg["model"]["res"]

    assert dtype in [torch.float64, torch.complex64], "dtype not implemented"
    
    assert downsampling_method in [
        "AvgPool",
        "MaxPool",
        "LPD",
        "APS",
        "LPF",
        None,
    ], "Downsampling method not implemented"
    assert upsampling_method in [
        "ConvTranspose",
        "Upsample",
        "LPU",
        None,
    ], "Upsampling method not implemented"
    assert normalization in [
        "BatchNorm",
        "LayerNorm",
        None,
    ], "Normalization method not implemented"

    assert model in [
        "AutoEncoder",
        "AutoEncoderWD",
        "UNet",
        "ResNet",
    ], "Model not implemented"

    if dtype == torch.float64:
        activation = eval(f"{activation}()", nn.modules.__dict__)
    elif dtype == torch.complex64:
        activation = eval(f"{activation}()", c_nn.__dict__)

    if isinstance(projection, str):
        projection = globals()[projection]()

    if model == "AutoEncoder":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=True,
            latent_dim=latent_dim,
            dropout=dropout,
            res=res,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            normalization_method=normalization,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "AutoEncoderWD":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=dropout,
            res=res,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            normalization_method=normalization,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "UNet":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=dropout,
            res=res,
            skip_connections=True,
            channel_attention=False,
            spatial_attention=False,
            normalization_method=normalization,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "ResNet":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=False,
            dense=True,
            latent_dim=None,
            dropout=dropout,
            res=res,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            normalization_method=normalization,
            downsampling_method=downsampling_method,
            upsampling_method=None,
            dtype=dtype,
        )


"""

# Local imports

from torchcvnn.nn.modules import activation as activation_cvnn
from torch.nn.modules import activation as activation_rvnn
from .learn_poly_sampling.layers import (
    PolyphaseInvariantDown2D,
    PolyphaseInvariantUp2D,
)
from .parts import *
from .model import DeepNeuralNetwork
from .projection import *


def build_model(cfg, projection, softmax, dtype, num_classes=None, num_channels=None):
    num_layers = cfg["model"]["num_layers"]
    channels_ratio = cfg["model"]["channels_ratio"]
    latent_dim = cfg["model"]["latent_dim"]
    img_size = cfg["data"]["img_size"]
    model = cfg["model"]["class"]
    activation = cfg["model"]["activation"]

    if dtype == torch.float64:
        activation = eval(f"{activation}()", activation_rvnn.__dict__)
    elif dtype == torch.complex64:
        activation = eval(f"{activation}()", activation_cvnn.__dict__)
    else:
        raise ValueError("dtype not supported")

    # activation = eval(  f"{activation}()",    )
    dropout = cfg["model"]["dropout"]
    if isinstance(projection, str):
        projection = globals()[projection]()

    assert model in [
        "AutoEncoder",
        "AutoEncoderRes",
        "AutoEncoderResAttention",
        "AutoEncoderWD",
        "AutoEncoderWDRes",
        "AutoEncoderWDResAttention",
        "AutoEncoderWDEquivariant",
        "AutoEncoderWDResEquivariant",
        "AutoEncoderWDResAttentionEquivariant",
        "UNet",
        "UNetRes",
        "UNetResAttention",
        "UNetEquivariant",
        "UNetResEquivariant",
        "UNetResAttentionEquivariant",
        "ResNet",
        "ResNetAttention",
        "ResNetEquivariant",
        "ResNetAttentionEquivariant",
    ], "Model not implemented"

    if model == "AutoEncoder":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=True,
            latent_dim=latent_dim,
            dropout=dropout,
            res=False,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderRes":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=True,
            latent_dim=latent_dim,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderResAttention":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=True,
            latent_dim=latent_dim,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderWD":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=False,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderWDRes":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderWDResAttention":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=False,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "AutoEncoderWDEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=False,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )

    elif model == "AutoEncoderWDResEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )

    elif model == "AutoEncoderWDResAttentionEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=None,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=False,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )

    elif model == "UNet":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=False,
            skip_connections=True,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "UNetRes":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=True,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "UNetResAttention":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=True,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "UNetEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=False,
            skip_connections=True,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )

    elif model == "UNetResEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=True,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )

    elif model == "UNetResAttentionEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=True,
            dense=False,
            latent_dim=None,
            dropout=None,
            res=True,
            skip_connections=True,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=PolyphaseInvariantUp2D,
            dtype=dtype,
        )
    elif model == "ResNet":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=False,
            dense=True,
            latent_dim=None,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "ResNetAttention":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=False,
            dense=True,
            latent_dim=None,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=None,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "ResNetEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=False,
            dense=True,
            latent_dim=None,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=False,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=None,
            dtype=dtype,
        )

    elif model == "ResNetAttentionEquivariant":
        return DeepNeuralNetwork(
            model=model,
            num_channels=num_channels,
            num_classes=num_classes,
            projection=projection,
            softmax=softmax,
            activation=activation,
            input_size=img_size,
            num_layers=num_layers,
            channels_ratio=channels_ratio,
            upsampling=False,
            dense=True,
            latent_dim=None,
            dropout=dropout,
            res=True,
            skip_connections=False,
            channel_attention=True,
            spatial_attention=False,
            downsampling_method=PolyphaseInvariantDown2D,
            upsampling_method=None,
            dtype=dtype,
        )
"""
