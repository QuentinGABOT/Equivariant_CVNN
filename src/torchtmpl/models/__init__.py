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


def build_model(
    cfg, projection, softmax, dtype, img_size, num_classes=None, num_channels=None
):
    num_layers = cfg["model"]["num_layers"]
    channels_ratio = cfg["model"]["channels_ratio"]
    latent_dim = cfg["model"]["latent_dim"]
    model = cfg["model"]["class"]
    activation = cfg["model"]["activation"]
    downsampling_method = cfg["model"]["downsampling"]
    upsampling_method = cfg["model"]["upsampling"]
    normalization = cfg["model"]["normalization"]["method"]
    dropout = cfg["model"]["dropout"]
    res = cfg["model"]["res"]
    track_running_stats = cfg["model"]["normalization"]["track_running_stats"]

    assert dtype in [torch.float64, torch.complex64], "dtype not implemented"

    assert downsampling_method in [
        "AvgPool",
        "MaxPool",
        "LPD",
        "APD",
        "LPF",
        "StridedConv",
        None,
    ], "Downsampling method not implemented"
    assert upsampling_method in [
        "ConvTranspose",
        "Upsample",
        "APU",
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
            normalization_method=normalization,
            track_running_stats=track_running_stats,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "AutoEncoderWD":
        return DeepNeuralNetwork(
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
            normalization_method=normalization,
            track_running_stats=track_running_stats,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "UNet":
        return DeepNeuralNetwork(
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
            normalization_method=normalization,
            track_running_stats=track_running_stats,
            downsampling_method=downsampling_method,
            upsampling_method=upsampling_method,
            dtype=dtype,
        )

    elif model == "ResNet":
        return DeepNeuralNetwork(
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
            latent_dim=latent_dim,
            dropout=dropout,
            res=res,
            skip_connections=False,
            normalization_method=normalization,
            track_running_stats=track_running_stats,
            downsampling_method=downsampling_method,
            upsampling_method=None,
            dtype=dtype,
        )
