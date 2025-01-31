import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .parts import DoubleConv, Down, Up, OutConv, Dense

DOWNSAMPLING_FACTOR = 2
UPSAMPLING_FACTOR = 2


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self,
        model,
        num_channels,
        num_layers,
        channels_ratio,
        input_size,
        activation,
        projection,
        dtype,
        softmax,
        normalization_method,
        downsampling_method,
        upsampling_method,
        dropout,
        latent_dim,
        res,
        num_classes=None,
        skip_connections=True,
        channel_attention=True,
        spatial_attention=True,
        upsampling=True,
        dense=False,
    ):
        super(DeepNeuralNetwork, self).__init__()
        assert dense or upsampling, "Either dense or upsampling must be provided"
        assert (
            num_classes is not None if dense and not upsampling else True
        ), "num_classes must be provided if dense and not upsampling"
        assert (
            upsampling_method is None if downsampling_method is None else True
        ), "upsampling_method can not be selected if downsampling_method is None"
        self.name = model
        self.n_channels = num_channels
        self.num_classes = num_classes
        self.upsampling = upsampling
        self.downsampling_method = downsampling_method
        self.upsampling_method = upsampling_method

        # Encoder with doubzling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.bridge_layers = []
        self.dense_layers = []
        self.decoder_layers = []

        self.encoder_layers.append(
            DoubleConv(
                in_channels=self.n_channels,
                out_channels=current_channels,
                activation=activation,
                normalization_method=normalization_method,
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
                        downsampling_method=downsampling_method,
                        projection=projection,
                        channel_attention=channel_attention,
                        spatial_attention=spatial_attention,
                        dtype=dtype,
                        softmax=softmax,
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
                            downsampling_method=downsampling_method,
                            projection=projection,
                            channel_attention=channel_attention,
                            spatial_attention=spatial_attention,
                            dtype=dtype,
                            softmax=softmax,
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
                            downsampling_method=downsampling_method,
                            projection=projection,
                            channel_attention=channel_attention,
                            spatial_attention=spatial_attention,
                            dtype=dtype,
                            softmax=softmax,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                        )
                    )
            input_size //= DOWNSAMPLING_FACTOR
            current_channels = out_channels

        self.encoder = nn.Sequential(*self.encoder_layers)

        self.bridge = nn.Sequential(*self.bridge_layers)

        if dense:
            self.dense_layers.append(
                Dense(
                    current_channels,
                    input_size,
                    num_classes,
                    latent_dim=latent_dim,
                    activation=activation,
                    dropout=dropout,
                    unflatten=upsampling,
                    projection=projection,
                    dtype=dtype,
                )
            )

        self.dense = nn.Sequential(*self.dense_layers)

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
                        skip_connections=skip_connections,
                        upsampling_method=upsampling_method,
                        projection=projection,
                        channel_attention=channel_attention,
                        spatial_attention=spatial_attention,
                        dtype=dtype,
                        softmax=softmax,
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
                        dropout=0,
                        dtype=dtype,
                    )
                )
            else:
                self.decoder_layers.append(
                    OutConv(
                        in_channels=current_channels,
                        out_channels=num_classes,
                        projection=projection,
                        dropout=dropout,
                        dtype=dtype,
                    )
                )

        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        probs = []
        skip_connections = []

        for enc in self.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
                probs.append(prob)
            else:
                x = enc(x)
            if self.bridge:
                skip_connections.append(x)

        if self.bridge:
            x, prob = self.bridge(x)
            probs.append(prob)

        if self.dense:
            x, x_projected = self.dense(x)

        skip_connections = skip_connections[::-1]
        probs = probs[::-1]

        if self.decoder:
            for idx, dec in enumerate(self.decoder):
                if isinstance(dec, Up):
                    if skip_connections:
                        skip = skip_connections[idx]
                    else:
                        skip = None
                    x = dec(x, skip, probs[idx])
                else:
                    x, x_projected = dec(x)

        return x, x_projected

    def use_checkpointing(self):
        for i, layer in enumerate(self.encoder):
            self.encoder[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.bridge):
            self.bridge[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.dense):
            self.dense[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.decoder):
            self.decoder[i] = checkpoint.checkpoint(layer)
