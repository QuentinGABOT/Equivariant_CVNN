import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .parts import DoubleConv, Down, Up, OutConv, Dense

DOWNSAMPLING_FACTOR = 2
UPSAMPLING_FACTOR = 2


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        num_layers,
        channels_ratio,
        input_size,
        activation,
        projection,
        dtype,
        softmax,
        normalization_method,
        track_running_stats,
        downsampling_method,
        upsampling_method,
        dropout,
        latent_dim,
        res,
        num_classes=None,
        skip_connections=True,
        upsampling=True,
        dense=False,
    ):
        super(DeepNeuralNetwork, self).__init__()

        assert dense or upsampling, "Either dense or upsampling must be provided"
        assert (
            num_classes is not None if dense and not upsampling else True
        ), "num_classes must be provided if dense and not upsampling"

        self.upsampling = upsampling
        self.skip_connections = skip_connections
        self.dense = dense

        # Encoder with doubzling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.bridge_layers = []
        self.dense_layers = []
        self.decoder_layers = []

        self.encoder_layers.append(
            DoubleConv(
                in_channels=num_channels,
                out_channels=current_channels,
                activation=activation,
                normalization_method=normalization_method,
                track_running_stats=track_running_stats,
                input_size=input_size,
                res=res,
                dtype=dtype,
            )
        )

        for i in range(1, num_layers + 1):
            out_channels = channels_ratio * 2**i
            if i < num_layers:
                self.encoder_layers.append(
                    Down(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        res=res,
                        input_size=input_size,
                        normalization_method=normalization_method,
                        track_running_stats=track_running_stats,
                        downsampling_method=downsampling_method,
                        projection=projection,
                        dtype=dtype,
                        softmax=softmax,
                        dropout=dropout,
                        downsampling_factor=DOWNSAMPLING_FACTOR,
                    )
                )
            else:
                if skip_connections:
                    self.bridge_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation,
                            res=res,
                            input_size=input_size,
                            normalization_method=normalization_method,
                            track_running_stats=track_running_stats,
                            downsampling_method=downsampling_method,
                            projection=projection,
                            dtype=dtype,
                            softmax=softmax,
                            dropout=dropout,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                        )
                    )
                else:
                    self.encoder_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation,
                            res=res,
                            input_size=input_size,
                            normalization_method=normalization_method,
                            track_running_stats=track_running_stats,
                            downsampling_method=downsampling_method,
                            projection=projection,
                            dtype=dtype,
                            softmax=softmax,
                            dropout=dropout,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                        )
                    )
            input_size //= DOWNSAMPLING_FACTOR
            current_channels = out_channels

        self.encoder_block = nn.Sequential(*self.encoder_layers)

        self.bridge_block = nn.Sequential(*self.bridge_layers)

        if dense:
            self.dense_layers.append(
                Dense(
                    current_channels,
                    input_size,
                    num_classes,
                    latent_dim=latent_dim,
                    activation=activation,
                    unflatten=upsampling,
                    projection=projection,
                    dtype=dtype,
                )
            )

        self.dense_block = nn.Sequential(*self.dense_layers)

        if upsampling:
            # Decoder with halving channels
            for i in range(num_layers - 1, -1, -1):
                out_channels = channels_ratio * 2**i
                self.decoder_layers.append(
                    Up(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        res=res,
                        input_size=input_size,
                        normalization_method=normalization_method,
                        track_running_stats=track_running_stats,
                        skip_connections=skip_connections,
                        upsampling_method=upsampling_method,
                        projection=projection,
                        dtype=dtype,
                        softmax=softmax,
                        dropout=dropout,
                        upsampling_factor=UPSAMPLING_FACTOR,
                    )
                )
                input_size *= UPSAMPLING_FACTOR
                current_channels = out_channels

            if num_classes is None:
                self.decoder_layers.append(
                    OutConv(
                        in_channels=current_channels,
                        out_channels=num_channels,
                        projection=projection,
                        dtype=dtype,
                    )
                )
            else:
                self.decoder_layers.append(
                    OutConv(
                        in_channels=current_channels,
                        out_channels=num_classes,
                        projection=projection,
                        dtype=dtype,
                    )
                )

        self.decoder_block = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        list_probs = []
        list_skip_connections = []

        for enc in self.encoder_block:
            if isinstance(enc, Down):
                x, prob = enc(x)
                list_probs.append(prob)
            else:
                x = enc(x)
            if self.skip_connections:
                list_skip_connections.append(x)

        if self.skip_connections:
            x, prob = self.bridge_block(x)
            list_probs.append(prob)

        if self.dense:
            x, x_projected = self.dense_block(x)

        list_skip_connections = list_skip_connections[::-1]
        list_probs = list_probs[::-1]

        if self.upsampling:
            for idx, dec in enumerate(self.decoder_block):
                if isinstance(dec, Up):
                    if self.skip_connections:
                        skip = list_skip_connections[idx]
                    else:
                        skip = None
                    x = dec(x, skip, list_probs[idx])
                else:
                    x, x_projected = dec(x)

        return x, x_projected

    def use_checkpointing(self):
        for i, layer in enumerate(self.encoder_block):
            self.encoder_block[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.bridge_block):
            self.bridge_block[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.dense_block):
            self.dense_block[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.decoder_block):
            self.decoder_block[i] = checkpoint.checkpoint(layer)
