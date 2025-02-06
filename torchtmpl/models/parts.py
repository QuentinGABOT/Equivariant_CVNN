import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn

from functools import partial
from .learn_poly_sampling.layers import (
    PolyphaseInvariantDown2D,
    get_logits_model,
    LPS,
    PolyphaseInvariantUp2D,
    LPS_u,
    Decimation,
    max_p_norm,
)
from .learn_poly_sampling.layers.lowpass_filter import LowPassFilter
from .learn_poly_sampling.layers.polydown import set_pool
from .learn_poly_sampling.layers.polyup import set_unpool
from ..models.projection import PolyCtoR, MLPCtoR, NoCtoR, ModCtoR
from .softmax import (
    Softmax,
    SoftmaxMeanCtoR,
    GumbelSoftmaxMeanCtoR,
    SoftmaxProductCtoR,
    GumbelSoftmaxProductCtoR,
    GumbelSoftmax,
)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        activation,
        dtype,
        normalization_method,
        track_running_stats,
        dropout=0,
        stride=1,
        res=False,
        mid_channels=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if stride == 1:
            padding = "same"  # padding='same' pads the input so the output has the shape as the input.
            # However, this mode doesn’t support any stride values other than 1.
        else:
            padding = 1

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
            dtype=self.dtype,
        )
        if normalization_method == "BatchNorm":
            if dtype == torch.complex64:
                self.normalization1 = c_nn.BatchNorm2d(
                    mid_channels, cdtype=dtype, track_running_stats=track_running_stats
                )
            elif dtype == torch.float64:
                self.normalization1 = nn.BatchNorm2d(
                    mid_channels, dtype=dtype, track_running_stats=track_running_stats
                )
        elif normalization_method == "LayerNorm":
            if dtype == torch.complex64:
                self.normalization1 = c_nn.LayerNorm(
                    normalized_shape=(mid_channels, input_size, input_size)
                )
            elif dtype == torch.float64:
                self.normalization1 = nn.LayerNorm(
                    normalized_shape=(mid_channels, input_size, input_size)
                )
        elif normalization_method == None:
            self.normalization1 = nn.Identity()

        self.activation = activation
        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            padding_mode="circular",
            dtype=self.dtype,
        )
        if normalization_method == "BatchNorm":
            if dtype == torch.complex64:
                self.normalization2 = c_nn.BatchNorm2d(
                    num_features=out_channels,
                    cdtype=dtype,
                    track_running_stats=track_running_stats,
                )
            elif dtype == torch.float64:
                self.normalization2 = nn.BatchNorm2d(
                    num_features=out_channels,
                    dtype=dtype,
                    track_running_stats=track_running_stats,
                )
        elif normalization_method == "LayerNorm":
            if dtype == torch.complex64:
                self.normalization2 = c_nn.LayerNorm(
                    normalized_shape=(out_channels, input_size, input_size)
                )
            elif dtype == torch.float64:
                self.normalization2 = nn.LayerNorm(
                    normalized_shape=(out_channels, input_size, input_size)
                )
        elif normalization_method == None:
            self.normalization2 = nn.Identity()

        if dtype == torch.complex64:
            self.dropout = c_nn.Dropout2d(dropout)
        elif dtype == torch.float64:
            self.dropout = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if (stride != 1 or in_channels != out_channels) and res:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding=0,
                    padding_mode="circular",
                    dtype=self.dtype,
                ),
                self.normalization2,
            )

    def forward(self, x):
        x = x.type(
            self.dtype
        )  # to ensure that the input is of type float and not double
        identity = x
        out = self.activation(self.normalization1(self.conv1(x)))

        out = self.normalization2(self.conv2(out))

        out = self.dropout(out)

        if self.res:
            if self.shortcut:
                identity = self.shortcut(identity)
            out += identity
        out = self.activation(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, projection):
        super(ChannelAttention, self).__init__()
        self.shared_layer_one = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            dtype=torch.float64,
        )
        if isinstance(projection, NoCtoR):
            self.projection = ModCtoR()
        else:
            self.projection = projection

        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        projected_input = self.projection(input_feature)
        # Global Average Pooling
        avg_pool = self.avg_pool(projected_input)
        avg_pool = avg_pool.squeeze(-1)  # Remove last dimension
        avg_pool = self.shared_layer_one(
            avg_pool
        )  # Conv1d expects input with shape [batch, channels, width]
        avg_pool = avg_pool.unsqueeze(-1)  # Add one dimension

        # Global Max Pooling
        max_pool = self.max_pool(projected_input)
        max_pool = max_pool.squeeze(-1)  # Remove last dimension
        max_pool = self.shared_layer_one(
            max_pool
        )  # Conv1d expects input with shape [batch, channels, width]
        max_pool = max_pool.unsqueeze(-1)  # Add one dimension

        # ECA Feature
        eca_feature = avg_pool + max_pool
        eca_feature = self.sigmoid(eca_feature)

        attention_feature = torch.mul(input_feature, eca_feature)
        return attention_feature


class SpatialAttention(
    nn.Module
):  # impossible to be shift-equivariant ? shift-invariant ?
    def __init__(self, projection, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=False,
            dtype=torch.float64,
        )
        if isinstance(projection, NoCtoR):
            self.projection = ModCtoR()
        else:
            self.projection = projection

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        projected_input = self.projection(input_feature)
        avg_pool = torch.mean(projected_input, dim=1, keepdim=True)
        max_pool, _ = torch.max(projected_input, dim=1, keepdim=True)
        pooled_feature = torch.cat([avg_pool, max_pool], dim=1)
        attention_feature = self.conv1(pooled_feature)
        attention_feature = self.sigmoid(attention_feature)
        attention_feature = torch.mul(input_feature, attention_feature).type(
            torch.complex64
        )
        return attention_feature


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        input_size,
        projection,
        softmax,
        normalization_method,
        track_running_stats,
        dtype,
        downsampling_factor,
        dropout,
        res=False,
        downsampling_method=None,
        channel_attention=False,
        spatial_attention=False,
        stride=1,
    ):
        super().__init__()

        if downsampling_method is None:
            self.downsampling_method = None
            stride = downsampling_factor
        elif downsampling_method == "LPD":
            self.downsampling_method = downsampling_lpd(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
                projection=projection,
                softmax=softmax,
                dtype=dtype,
                downsampling_factor=downsampling_factor,
            )
        elif downsampling_method == "APS":
            self.downsampling_method = downsampling_aps(
                in_channels=in_channels,
                out_channels=out_channels,
                downsampling_factor=downsampling_factor,
            )
        elif downsampling_method == "LPF":
            self.downsampling_method = downsampling_lpf(
                in_channels=in_channels,
                out_channels=out_channels,
                downsampling_factor=downsampling_factor,
            )
        elif downsampling_method == "MaxPool":
            if dtype == torch.complex64:
                self.downsampling_method = c_nn.MaxPool2d(downsampling_factor)
            elif dtype == torch.float64:
                self.downsampling_method = nn.MaxPool2d(downsampling_factor)
        elif downsampling_method == "AvgPool":
            if dtype == torch.complex64:
                self.downsampling_method = c_nn.AvgPool2d(
                    downsampling_factor, stride=downsampling_factor
                )
            elif dtype == torch.float64:
                self.downsampling_method = nn.AvgPool2d(downsampling_factor)

        input_size = input_size // downsampling_factor

        if channel_attention:
            self.channel_attention = ChannelAttention(
                out_channels, projection=projection
            )
        else:
            self.channel_attention = None
        if spatial_attention:
            self.spatial_attention = SpatialAttention(projection=projection)
        else:
            self.spatial_attention = None
        self.conv_layer = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            input_size=input_size,
            normalization_method=normalization_method,
            track_running_stats=track_running_stats,
            dtype=dtype,
            stride=stride,
            res=res,
            dropout=dropout,
        )

    def forward(self, x):
        prob = None
        if self.downsampling_method is not None:
            if isinstance(self.downsampling_method, PolyphaseInvariantDown2D):
                x, prob = self.downsampling_method(x, ret_prob=True)
            elif isinstance(
                self.downsampling_method,
                (c_nn.MaxPool2d, nn.MaxPool2d, c_nn.AvgPool2d, nn.AvgPool2d),
            ):
                x = self.downsampling_method(x)
        x = self.conv_layer(x)
        if self.channel_attention:
            x = self.channel_attention(x)
        if self.spatial_attention:
            x = self.spatial_attention(x)
        return x, prob


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        projection,
        input_size,
        normalization_method,
        track_running_stats,
        dtype,
        softmax,
        upsampling_factor,
        dropout,
        res=False,
        upsampling_method=None,
        skip_connections=False,
        channel_attention=False,
        spatial_attention=False,
    ):
        super().__init__()
        if upsampling_method == "Upsample":
            if dtype == torch.complex64:
                self.upsampling_method = c_nn.Upsample(
                    scale_factor=upsampling_factor, mode="bilinear"
                )
            elif dtype == torch.float64:
                self.upsampling_method = nn.Upsample(
                    scale_factor=upsampling_factor, mode="bilinear"
                )

        elif upsampling_method == "ConvTranspose":
            if dtype == torch.complex64:
                self.upsampling_method = c_nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=upsampling_factor
                )
            elif dtype == torch.float64:
                self.upsampling_method = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=upsampling_factor,
                    dtype=dtype,
                )
            in_channels = out_channels

        elif upsampling_method == "LPU":
            self.upsampling_method = upsampling_lpu(
                in_channels, softmax=softmax, upsampling_factor=upsampling_factor
            )

        input_size = input_size * upsampling_factor

        if channel_attention:
            self.channel_attention = ChannelAttention(
                out_channels, projection=projection
            )
        else:
            self.channel_attention = None
        if spatial_attention:
            self.spatial_attention = SpatialAttention(projection=projection)
        else:
            self.spatial_attention = None

        if skip_connections:
            in_channels += out_channels

        self.conv_layer = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            input_size=input_size,
            normalization_method=normalization_method,
            track_running_stats=track_running_stats,
            res=res,
            dtype=dtype,
            dropout=dropout,
        )

    def forward(self, x1, x2=None, prob=None):
        if isinstance(
            self.upsampling_method,
            (c_nn.ConvTranspose2d, nn.ConvTranspose2d, nn.Upsample, c_nn.Upsample),
        ):
            x1 = self.upsampling_method(x1)
        elif isinstance(self.upsampling_method, PolyphaseInvariantUp2D):
            x1 = self.upsampling_method(x1, prob=prob)
        x = concat(x1, x2)
        x = self.conv_layer(x)
        if self.channel_attention:
            x = self.channel_attention(x)
        if self.spatial_attention:
            x = self.spatial_attention(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, projection, dtype):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                dtype=dtype,
            ),
        )
        self.projection = projection

    def forward(self, x):
        x = self.conv(x)
        x_projected = self.projection(x)
        return x, x_projected


class Dense(nn.Module):
    def __init__(
        self,
        in_channels,
        input_size,
        num_classes,
        projection,
        dtype,
        latent_dim=None,
        activation=None,
        unflatten: bool = False,
    ):
        super(Dense, self).__init__()
        if dtype == torch.complex64:
            self.avg_pool = c_nn.AvgPool2d(input_size, input_size)
        elif dtype == torch.float64:
            self.avg_pool = nn.AvgPool2d(input_size, input_size)
        self.unflatten = unflatten

        if activation:
            self.activation = activation
        else:
            self.activation = None

        self.projection = projection

        if unflatten:
            assert latent_dim is not None
            linear = in_channels * input_size * input_size
            self.fc_1 = nn.Linear(
                in_features=in_channels, out_features=latent_dim, dtype=dtype
            )
            self.fc_2 = nn.Linear(
                in_features=latent_dim, out_features=linear, dtype=dtype
            )
            self.unflat_1 = nn.Unflatten(
                dim=1, unflattened_size=(in_channels, input_size, input_size)
            )
        else:
            self.fc_1 = nn.Linear(
                in_features=in_channels, out_features=num_classes, dtype=dtype
            )

    def forward(self, x):
        x_projected = None
        x = torch.flatten(self.avg_pool(x), 1)
        x = self.fc_1(x)

        if self.unflatten:
            if self.activation:
                x = self.activation(x)
            x = self.fc_2(x)
            x = self.unflat_1(x)
        else:
            x_projected = self.projection(x)
        return x, x_projected


def downsampling_lpd(
    in_channels,
    out_channels,
    activation,
    projection,
    softmax,
    dtype,
    downsampling_factor,
):
    if isinstance(softmax, SoftmaxMeanCtoR):
        gumbel_softmax = GumbelSoftmaxMeanCtoR()
    elif isinstance(softmax, SoftmaxProductCtoR):
        gumbel_softmax = GumbelSoftmaxProductCtoR()
    elif isinstance(softmax, Softmax):
        gumbel_softmax = GumbelSoftmax()
    else:
        raise ValueError("Unexpected softmax layer")
    return set_pool(
        partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model("LPSLogitLayers"),
            pass_extras=False,
            no_antialias=True,
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        projection=projection,
        activation=activation,
        gumbel_softmax=gumbel_softmax,
        softmax=softmax,
        dtype=dtype,
        stride=downsampling_factor,
    )


def downsampling_aps(in_channels, out_channels, downsampling_factor):
    return set_pool(
        partial(
            PolyphaseInvariantDown2D,
            component_selection=max_p_norm,
            get_logits=get_logits_model("LPSLogitLayers"),
            pass_extras=False,
            no_antialias=True,
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        stride=downsampling_factor,
    )


def downsampling_lpf(in_channels, out_channels, downsampling_factor):
    return set_pool(
        partial(
            Decimation,
            antialias_layer=partial(
                LowPassFilter, filter_size=3, padding="valid", padding_mode="circular"
            ),
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        no_antialias=False,
        stride=downsampling_factor,
    )


def upsampling_lpu(in_channels, softmax, upsampling_factor):
    return set_unpool(
        partial(PolyphaseInvariantUp2D, component_selection=LPS_u),
        p_ch=in_channels,
        no_antialias=True,
        softmax=softmax,
        stride=upsampling_factor,
    )


def concat(x1, x2):
    if x2 is None:
        return x1
    else:
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
