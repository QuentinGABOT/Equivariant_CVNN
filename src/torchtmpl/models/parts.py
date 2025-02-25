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
    max_p_norm_u,
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
        elif downsampling_method == "StridedConv":
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
                no_antialias=True,
            )
        elif downsampling_method == "LPD_F":
            self.downsampling_method = downsampling_lpd(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
                projection=projection,
                softmax=softmax,
                dtype=dtype,
                downsampling_factor=downsampling_factor,
                no_antialias=False,
            )
        elif downsampling_method == "APD":
            self.downsampling_method = downsampling_apd(
                in_channels=in_channels,
                out_channels=out_channels,
                downsampling_factor=downsampling_factor,
                no_antialias=True,
            )
        elif downsampling_method == "APD_F":
            self.downsampling_method = downsampling_apd(
                in_channels=in_channels,
                out_channels=out_channels,
                downsampling_factor=downsampling_factor,
                no_antialias=False,
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

        if self.downsampling_method is not None:
            if isinstance(self.downsampling_method, PolyphaseInvariantDown2D):
                x, prob = self.downsampling_method(x, ret_prob=True)
            else:
                prob = None
                x = self.downsampling_method(x)
        else:
            prob = None
        x = self.conv_layer(x)

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
        elif upsampling_method == "APU":
            self.upsampling_method = upsampling_apu(
                in_channels, upsampling_factor=upsampling_factor, no_antialias=True
            )
        elif upsampling_method == "APU_F":
            self.upsampling_method = upsampling_apu(
                in_channels, upsampling_factor=upsampling_factor, no_antialias=False
            )
        elif upsampling_method == "LPU":
            self.upsampling_method = upsampling_lpu(
                in_channels,
                softmax=softmax,
                upsampling_factor=upsampling_factor,
                no_antialias=True,
            )
        elif upsampling_method == "LPU_F":
            self.upsampling_method = upsampling_lpu(
                in_channels,
                softmax=softmax,
                upsampling_factor=upsampling_factor,
                no_antialias=False,
            )
        input_size = input_size * upsampling_factor

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
        if isinstance(self.upsampling_method, PolyphaseInvariantUp2D):
            x1 = self.upsampling_method(x1, prob=prob)
        else:
            x1 = self.upsampling_method(x1)
        x = concat(x1, x2)
        x = self.conv_layer(x)

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
        activation,
        latent_dim,
        unflatten: bool = False,
    ):
        super(Dense, self).__init__()
        if dtype == torch.complex64:
            self.avg_pool = c_nn.AvgPool2d(input_size, input_size)
        elif dtype == torch.float64:
            self.avg_pool = nn.AvgPool2d(input_size, input_size)

        self.unflatten = unflatten
        self.activation = activation
        self.projection = projection

        if unflatten:
            out_features = in_channels * input_size * input_size
            self.unflat = nn.Unflatten(
                dim=1, unflattened_size=(in_channels, input_size, input_size)
            )
        else:
            out_features = num_classes

        self.fc_1 = nn.Linear(
            in_features=in_channels, out_features=latent_dim, dtype=dtype
        )
        self.fc_2 = nn.Linear(
            in_features=latent_dim, out_features=out_features, dtype=dtype
        )

    def forward(self, x):
        x = torch.flatten(self.avg_pool(x), 1)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)

        if self.unflatten:
            x = self.unflat(x)
            x_projected = None
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
    no_antialias,
):
    if isinstance(softmax, SoftmaxMeanCtoR):
        gumbel_softmax = GumbelSoftmaxMeanCtoR()
    elif isinstance(softmax, SoftmaxProductCtoR):
        gumbel_softmax = GumbelSoftmaxProductCtoR()
    elif isinstance(softmax, Softmax):
        gumbel_softmax = GumbelSoftmax()
    else:
        raise ValueError("Unexpected softmax layer")
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter, filter_size=3, padding="same", padding_mode="circular"
        )
    return set_pool(
        partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model("LPSLogitLayers"),
            pass_extras=False,
            antialias_layer=antialias_layer,
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        projection=projection,
        activation=activation,
        gumbel_softmax=gumbel_softmax,
        softmax=softmax,
        dtype=dtype,
        stride=downsampling_factor,
        no_antialias=no_antialias,
    )


def downsampling_apd(in_channels, out_channels, downsampling_factor, no_antialias):
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter, filter_size=3, padding="same", padding_mode="circular"
        )

    return set_pool(
        partial(
            PolyphaseInvariantDown2D,
            component_selection=max_p_norm,
            get_logits=get_logits_model("LPSLogitLayers"),
            pass_extras=False,
            antialias_layer=antialias_layer,
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        stride=downsampling_factor,
        no_antialias=no_antialias,
    )


def downsampling_lpf(in_channels, out_channels, downsampling_factor):
    return set_pool(
        partial(
            Decimation,
            antialias_layer=partial(
                LowPassFilter, filter_size=3, padding="same", padding_mode="circular"
            ),
        ),
        p_ch=in_channels,
        h_ch=out_channels,
        no_antialias=False,
        stride=downsampling_factor,
    )


def upsampling_apu(in_channels, upsampling_factor, no_antialias):
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter, filter_size=3, padding="same", padding_mode="circular"
        )
    return set_unpool(
        partial(
            PolyphaseInvariantUp2D,
            component_selection=max_p_norm_u,
            antialias_layer=antialias_layer,
        ),
        p_ch=in_channels,
        no_antialias=no_antialias,
        stride=upsampling_factor,
        softmax=None,
    )


def upsampling_lpu(in_channels, softmax, upsampling_factor, no_antialias):
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter, filter_size=3, padding="same", padding_mode="circular"
        )
    return set_unpool(
        partial(
            PolyphaseInvariantUp2D,
            component_selection=LPS_u,
            antialias_layer=antialias_layer,
        ),
        p_ch=in_channels,
        no_antialias=no_antialias,
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
