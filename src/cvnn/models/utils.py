"""
Mode-aware utilities for handling real vs complex-valued model components.

This module provides centralized functions for selecting appropriate
components (activations, losses, normalizations, initializations) based
on the layer_mode parameter. It ensures seamless switching between
PyTorch and torchcvnn components while maintaining API consistency.

Key Functions:
    is_real_mode: Check if layer mode uses real arithmetic
    get_activation: Get mode-appropriate activation function
    get_loss_function: Get mode-appropriate loss function
    get_normalization: Get mode-appropriate normalization layer
    init_weights_mode_aware: Initialize weights appropriately for mode
    validate_layer_mode: Validate layer mode parameter

Usage Example:
    >>> from cvnn.models.mode_utils import get_activation, is_real_mode
    >>> activation = get_activation("ReLU", layer_mode="complex")
    >>> is_real = is_real_mode("real")
"""

# Standard library imports
import warnings
from typing import Optional, Union, Dict, Any, Tuple
from functools import partial

# Third-party imports
import torch
import numpy as np
import torch.nn as nn
import torchcvnn.nn.modules as c_nn

# Local imports
from .projection import PolyCtoR, MLPCtoR
from .softmax import (
    GumbelSoftmax,
    GumbelSoftmaxMeanCtoR,
    GumbelSoftmaxProductCtoR,
    Softmax,
    SoftmaxMeanCtoR,
    SoftmaxProductCtoR,
)
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
import cvnn.losses as custom_losses


def is_real_mode(layer_mode: str) -> bool:
    """Check if the layer mode corresponds to real-valued operations.

    Real modes use standard PyTorch components, while complex modes
    use torchcvnn components that handle complex arithmetic.

    Args:
        layer_mode (str): The layer mode to check. Must be one of:
            - "real": Standard real-valued neural networks
            - "complex": Full complex-valued networks
            - "split": Split complex representations

    Returns:
        bool: True if mode uses real-valued operations (real),
              False if complex-valued operations (complex/split)

    Raises:
        ValueError: If layer_mode is not one of the supported modes

    Example:
        >>> is_real_mode("real")
        True
        >>> is_real_mode("complex")
        False
    """
    if layer_mode == "real":
        return True
    elif layer_mode in ["complex", "split"]:
        return False
    else:
        raise ValueError(
            f"Invalid layer_mode '{layer_mode}'. Must be one of: real, complex, split"
        )


def get_activation(activation_name: str, layer_mode: str) -> nn.Module:
    """Get appropriate activation function based on layer mode.

    Automatically selects between PyTorch and torchcvnn activations
    based on the layer mode. Uses generic names that work across modes.

    Args:
        activation_name (str): Generic activation name. Supported:
            - "ReLU": Maps to ReLU (real) or modReLU (complex)
            - "Tanh": Maps to Tanh (real) or Tanh (complex)
            - "Sigmoid": Maps to Sigmoid (real) or Sigmoid (complex)
            - "LeakyReLU": Maps to LeakyReLU (real) or LeakyReLU (complex)
            - "CReLU": Only available for complex modes
        layer_mode (str): The layer mode (real/complex/split)

    Returns:
        nn.Module: Instantiated activation module appropriate for the mode

    Raises:
        ValueError: If activation_name is unknown or incompatible with layer_mode

    Example:
        >>> # Real mode gets torch.nn.ReLU
        >>> activation = get_activation("ReLU", layer_mode="real")
        >>> # Complex mode gets torchcvnn.nn.modules.modReLU
        >>> activation = get_activation("ReLU", layer_mode="complex")
    """
    # Mapping from generic/complex activation names to real equivalents
    COMPLEX_TO_REAL_ACTIVATIONS = {
        "modReLU": "ReLU",
        "CReLU": "ReLU",
        "zReLU": "ReLU",
        "modTanh": "Tanh",
        "modSigmoid": "Sigmoid",
        "modELU": "ELU",
        "modLeakyReLU": "LeakyReLU",
        # Direct mappings for standard activations
        "ReLU": "ReLU",
        "Tanh": "Tanh",
        "Sigmoid": "Sigmoid",
        "ELU": "ELU",
        "LeakyReLU": "LeakyReLU",
        "GELU": "GELU",
        "Swish": "SiLU",  # Swish is SiLU in PyTorch
        "Mish": "Mish",
    }
    if activation_name is None:
        return nn.Identity()

    if is_real_mode(layer_mode):
        # For real modes, use PyTorch activations
        if activation_name in COMPLEX_TO_REAL_ACTIVATIONS:
            real_activation_name = COMPLEX_TO_REAL_ACTIVATIONS[activation_name]
            try:
                activation_cls = getattr(nn, real_activation_name)
                return activation_cls()
            except AttributeError:
                raise ValueError(
                    f"Real activation '{real_activation_name}' not found in torch.nn. "
                    f"Available activations: {list(COMPLEX_TO_REAL_ACTIVATIONS.values())}"
                )
        else:
            # Unknown activation - check if it's available in real form
            try:
                activation_cls = getattr(nn, activation_name)
                warnings.warn(
                    f"Using real activation '{activation_name}' directly for real mode '{layer_mode}'. "
                    f"Consider using a mapped activation from: {list(COMPLEX_TO_REAL_ACTIVATIONS.keys())}"
                )
                return activation_cls()
            except AttributeError:
                raise ValueError(
                    f"Activation '{activation_name}' not found for real mode '{layer_mode}'. "
                    f"Available mappings: {list(COMPLEX_TO_REAL_ACTIVATIONS.keys())} "
                    f"or standard PyTorch activations."
                )
    else:
        # For complex modes, use torchcvnn activations
        try:
            activation_cls = getattr(c_nn, activation_name)
            return activation_cls()
        except AttributeError:
            try:
                activation_cls = getattr(nn, activation_name)
                return activation_cls()
            except AttributeError:
                # Check if we can map to a complex activation
                if activation_name in COMPLEX_TO_REAL_ACTIVATIONS:
                    # This is a standard activation, but user wants complex mode
                    warnings.warn(
                        f"Activation '{activation_name}' may not be available in torchcvnn. "
                        f"For complex mode '{layer_mode}', consider using complex activations like 'modReLU'."
                    )
                raise ValueError(
                    f"Complex activation '{activation_name}' not found in torchcvnn. "
                    f"Available complex activations depend on your torchcvnn installation."
                )


def get_loss_function(
    loss_name: str,
    layer_mode: str,
    ignore_index: Optional[int] = None,
    class_weights: Optional[np.ndarray] = None,
    schedule: Optional[str] = None,
    bpd_target: float = 0.20,
    beta_max: float = 1.0,
    cov_mode: str = "diag",
    standard_reparam: bool = False,
    learned_variance: bool = False,
    min_log_sigma: Optional[float] = None,
) -> nn.Module:
    """Get appropriate loss function based on layer mode.

    Automatically selects between PyTorch and torchcvnn loss functions
    based on the layer mode. Uses generic names for consistency.

    Args:
        loss_name (str): Generic loss function name. Supported:
            - "MSE": Mean Squared Error loss
            - "L1": L1/Mean Absolute Error loss
            - "CrossEntropy": Cross-entropy loss
            - "BCE": Binary Cross-Entropy loss
            - "BCEWithLogits": BCE with logits loss
            - "Huber": Huber/Smooth L1 loss
            - "FocalLoss": Focal loss for class imbalance (custom implementation)
        layer_mode (str): The layer mode (real/complex/split)
        ignore_index (Optional[int]): Index to ignore for losses that support it (e.g., CrossEntropy, FocalLoss)

    Returns:
        nn.Module: Instantiated loss function appropriate for the mode

    Raises:
        ValueError: If loss_name is unknown or incompatible with layer_mode

    Example:
        >>> # Real mode gets torch.nn.MSELoss
        >>> loss = get_loss_function("MSE", layer_mode="real")
        >>> # Complex mode gets torchcvnn.nn.modules.ComplexMSELoss
        >>> loss = get_loss_function("MSE", layer_mode="complex")
        >>> # CrossEntropy with ignore_index
        >>> loss = get_loss_function("CrossEntropy", layer_mode="real", ignore_index=0)
        >>> # FocalLoss for class imbalance
        >>> loss = get_loss_function("FocalLoss", layer_mode="real", ignore_index=-100)

    Note:
        Some losses (like CrossEntropy) may only be available for certain modes.
        The function will raise an error for incompatible combinations.
    """
    # Mapping from generic loss names to specific implementations
    LOSS_MAPPINGS = {
        "MSE": {"complex": "ComplexMSELoss", "real": "MSELoss"},
        "L1": {"complex": "ComplexL1Loss", "real": "L1Loss"},
        "CrossEntropy": {
            "complex": "CrossEntropyLoss",
            "real": "CrossEntropyLoss",
        },
        "BCE": {"complex": "ComplexBCELoss", "real": "BCELoss"},
        "BCEWithLogits": {
            "complex": "ComplexBCEWithLogitsLoss",
            "real": "BCEWithLogitsLoss",
        },
        "Huber": {"complex": "ComplexHuberLoss", "real": "HuberLoss"},
        "FocalLoss": {"complex": "FocalLoss", "real": "FocalLoss"},
        "ELBOLoss": {"complex": "ComplexELBOLoss", "real": "ELBOLoss"},
    }

    if loss_name not in LOSS_MAPPINGS:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available losses: {list(LOSS_MAPPINGS.keys())}"
        )

    loss_mapping = LOSS_MAPPINGS[loss_name]

    if is_real_mode(layer_mode):
        # Use PyTorch loss functions for real modes
        real_loss_name = loss_mapping["real"]
        try:
            loss_cls = getattr(nn, real_loss_name)
            # Pass ignore_index to losses that support it
            if ignore_index is not None and loss_name in ["CrossEntropy"]:
                return loss_cls(ignore_index=ignore_index, weight=class_weights)
            else:
                return loss_cls()
        except AttributeError:
            # Try to find in custom losses module
            try:
                loss_cls = getattr(custom_losses, real_loss_name)
                # Pass ignore_index to losses that support it
                if ignore_index is not None and loss_name in [
                    "FocalLoss"
                ]:
                    return loss_cls(ignore_index=ignore_index, weight=class_weights)
                elif loss_name in ["ELBOLoss"]:
                    return loss_cls(schedule=schedule, bpd_target=bpd_target, beta_max=beta_max, cov_mode=cov_mode,
                                    learned_variance=learned_variance, min_log_sigma=min_log_sigma)
            except (ImportError, AttributeError):
                raise ValueError(
                    f"Real loss '{real_loss_name}' not found in torch.nn or cvnn.losses"
                )
    else:
        # Use torchcvnn loss functions for complex modes
        complex_loss_name = loss_mapping["complex"]
        try:
            loss_cls = getattr(c_nn, complex_loss_name)
            # Pass ignore_index to complex losses that support it
            if ignore_index is not None and loss_name in ["CrossEntropy", "FocalLoss"]:
                return loss_cls(ignore_index=ignore_index, weight=class_weights)
            else:
                return loss_cls()
        except AttributeError:
            # Try to find in custom losses module
            try:
                loss_cls = getattr(custom_losses, complex_loss_name)
                # Pass ignore_index to losses that support it
                if ignore_index is not None and loss_name in [
                    "FocalLoss",
                ]:
                    return loss_cls(ignore_index=ignore_index, weight=class_weights)
                elif loss_name in ["ELBOLoss"]:
                    return loss_cls(schedule=schedule, bpd_target=bpd_target, beta_max=beta_max, cov_mode=cov_mode,
                                    standard_reparam=standard_reparam, learned_variance=learned_variance, min_log_sigma=min_log_sigma)
                else:
                    return loss_cls()
            except (ImportError, AttributeError):
                # Try to suggest alternative or provide helpful error
                available_losses = [name for name in dir(c_nn) if "Loss" in name]
                raise ValueError(
                    f"Complex loss '{complex_loss_name}' not found in torchcvnn or cvnn.losses. "
                    f"Available complex losses in torchcvnn: {available_losses}"
                )


def get_normalization(
    norm_type: Optional[str],
    layer_mode: Optional[str],
    num_features: int,
    dimensionality: Optional[int] = 2,
) -> nn.Module:
    """Get appropriate normalization layer based on layer mode.

    Automatically selects between PyTorch and torchcvnn normalization
    layers based on the layer mode and input tensor requirements.

    Args:
        norm_type (str): Type of normalization. Supported:
            - "batch": Batch normalization
            - "layer": Layer normalization
            - "instance": Instance normalization
            - "group": Group normalization
            - "none" or None: No normalization (returns Identity)
        layer_mode (str): The layer mode (real/complex/split)
        num_features (int): Number of features/channels for normalization
        dimensionality (int): Dimensionality of the input (e.g., 1 for 1D, 2 for 2D)

    Returns:
        nn.Module: Instantiated normalization layer appropriate for the mode

    Raises:
        ValueError: If norm_type is unknown

    Example:
        >>> # Real mode gets torch.nn.BatchNorm2d
        >>> norm = get_normalization("batch", "real", num_features=64)
        >>> # Complex mode gets torchcvnn.nn.modules.BatchNorm2d
        >>> norm = get_normalization("batch", "complex", num_features=64)

    Note:
        The function automatically handles the different parameter names
        and requirements between PyTorch and torchcvnn normalization layers.
    """

    if num_features is None or num_features <= 0:
        raise ValueError(
            f"num_features is required for normalization type '{norm_type}'"
        )

    # Coerce optional inputs
    layer_mode = layer_mode or "complex"
    if norm_type is None or (isinstance(norm_type, str) and norm_type.lower() == "none"):
        return nn.Identity()
    
    # Mapping from normalization types to implementations
    NORM_MAPPINGS = {
        "batch": {
            "complex": lambda: c_nn.BatchNorm2d(num_features=num_features) if dimensionality == 2 else c_nn.BatchNorm1d(num_features=num_features, affine=True) if dimensionality ==1 else None,
            "real": lambda: nn.BatchNorm2d(num_features=num_features) if dimensionality == 2 else nn.BatchNorm1d(num_features=num_features, affine=True) if dimensionality ==1 else None,
        },
        "layer": {
            "complex": lambda: c_nn.LayerNorm(
                normalized_shape=(num_features,)
            ),
            "real": lambda: nn.GroupNorm(
                            num_groups=1, 
                            num_channels=num_features
                        ),       
        },
        "instance": {
            "complex": lambda: c_nn.InstanceNorm2d(num_features=num_features) if dimensionality == 2 else c_nn.LayerNorm(normalized_shape=(num_features,)) if dimensionality ==1 else None,
            "real": lambda: nn.InstanceNorm2d(num_features=num_features) if dimensionality == 2 else nn.LayerNorm(normalized_shape=(num_features,)) if dimensionality ==1 else None,
        },
        "group": {
            "complex": lambda: c_nn.GroupNorm(
                num_groups=max(1, num_features // 8), 
                num_channels=num_features
            ),
            "real": lambda: nn.GroupNorm(
                num_groups=max(1, num_features // 8), num_channels=num_features
            ),
        },
    }

    norm_type_lower = (norm_type or "").lower()
    if norm_type_lower not in NORM_MAPPINGS:
        raise ValueError(
            f"Unknown normalization type '{norm_type}'. "
            f"Available types: {list(NORM_MAPPINGS.keys())}"
        )

    norm_mapping = NORM_MAPPINGS[norm_type_lower]
    if is_real_mode(layer_mode):
        return norm_mapping["real"]()
    else:
        return norm_mapping["complex"]()


def get_softmax(
    gumbel_softmax_type: Optional[str], layer_mode: str = "complex"
) -> Tuple[Any, Any]:
    """Get appropriate Gumbel softmax layer based on layer mode.

    Automatically selects between PyTorch and torchcvnn Gumbel softmax
    implementations based on the layer mode.

    Args:
        gumbel_softmax_type (str): Type of Gumbel softmax. Supported:
            - "mean": GumbelSoftmaxMeanCtoR
            - "product": GumbelSoftmaxProductCtoR
            - "standard": GumbelSoftmax
        layer_mode (str): The layer mode (real/complex/split)

    Returns:
        Tuple[softmax_layer, gumbel_softmax_layer]: A pair where the first
        element is the selected softmax wrapper and the second is the
        GumbelSoftmax-like module. Types depend on the mode and installed
        optional packages.

    Raises:
        ValueError: If softmax_type is unknown or incompatible with layer_mode

    Example:
        >>> # Complex mode gets GumbelSoftmaxMeanCtoR
        >>> gumbel_softmax = get_gumbel_softmax("mean", layer_mode="complex")
    """
    if is_real_mode(layer_mode):
        return Softmax(), GumbelSoftmax()  # Real mode uses standard GumbelSoftmax
    else:
        if gumbel_softmax_type == "mean":
            return SoftmaxMeanCtoR(), GumbelSoftmaxMeanCtoR()
        elif gumbel_softmax_type == "product":
            return SoftmaxProductCtoR(), GumbelSoftmaxProductCtoR()
        elif gumbel_softmax_type == None or gumbel_softmax_type == "standard":
            return Softmax(), GumbelSoftmax()
        else:
            raise ValueError(
                f"Unknown softmax type '{gumbel_softmax_type}' for complex mode '{layer_mode}'. "
                f"Available types: mean, product, standard"
            )


def get_downsampling(
    downsampling: Optional[str],
    projection: Optional[str] = "amplitude",
    projection_config: Optional[Dict[str, Any]] = None,
    layer_mode: str = "complex",
    factor: int = 2,
    gumbel_softmax_type: Optional[str] = None,
    conv: Optional[nn.Module] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
) -> nn.Module:
    """Retrieve downsampling layer by name with mode awareness.

    Args:
        downsampling: Downsampling method (e.g., "maxpool", "avgpool", "none")
        projection: Projection method for LPD variants
        projection_config: Configuration for projection layer
        layer_mode: Layer mode to determine real vs complex implementation
        factor: Downsampling factor
        gumbel_softmax_type: Type of Gumbel softmax for LPD variants. If None and
            projection is None, defaults to "mean". Otherwise defaults to "standard".
        conv: Optional convolutional layer
        in_channels: Input channels
        out_channels: Output channels

    Returns:
        Instantiated downsampling module
    """

    if downsampling is None or downsampling == "none":
        return nn.Identity()

    is_real = is_real_mode(layer_mode)

    if downsampling == "maxpool":
        return (
            nn.MaxPool2d(kernel_size=factor, stride=factor)
            if is_real
            else c_nn.MaxPool2d(kernel_size=factor, stride=factor)
        )
    elif downsampling == "avgpool":
        return (
            nn.AvgPool2d(kernel_size=factor, stride=factor, count_include_pad=False)
            if is_real
            else c_nn.AvgPool2d(
                kernel_size=factor, stride=factor, count_include_pad=False
            )
        )
    elif downsampling == "LPD":
        return downsampling_lpd(
            conv=conv,
            downsampling_factor=factor,
            gumbel_softmax_type=gumbel_softmax_type,
            projection=projection,
            projection_config=projection_config,
            layer_mode=layer_mode,
            no_antialias=True,
        )
    elif downsampling == "LPD_F":
        return downsampling_lpd(
            conv=conv,
            downsampling_factor=factor,
            gumbel_softmax_type=gumbel_softmax_type,
            no_antialias=False,
            projection=projection,
            projection_config=projection_config,
            layer_mode=layer_mode,
            in_channels=in_channels,
        )
    elif downsampling == "APD":
        return downsampling_apd(
            in_channels=in_channels,
            downsampling_factor=factor,
            no_antialias=True,
            layer_mode=layer_mode,
        )
    elif downsampling == "APD_F":
        return downsampling_apd(
            in_channels=in_channels,
            downsampling_factor=factor,
            no_antialias=False,
            layer_mode=layer_mode,
        )
    elif downsampling == "LPF":
        return downsampling_lpf(
            in_channels=in_channels,
            downsampling_factor=factor,
            layer_mode=layer_mode,
        )
    else:
        raise ValueError(f"Unsupported downsampling method: {downsampling}")


def downsampling_lpd(
    conv,
    downsampling_factor,
    no_antialias,
    gumbel_softmax_type: Optional[str],
    projection: Optional[str],
    projection_config,
    layer_mode,
    in_channels=None,
):
    # Determine effective softmax type based on projection
    projection_layer = get_projection(
        projection, layer_mode=layer_mode, projection_config=projection_config
    )

    # If projection is None and gumbel_softmax_type is None, default to "mean"
    # Otherwise use provided gumbel_softmax_type (None defaults to "standard")
    effective_softmax_type = gumbel_softmax_type
    if projection is None and gumbel_softmax_type is None:
        effective_softmax_type = "mean"

    softmax, gumbel_softmax = get_softmax(effective_softmax_type, layer_mode=layer_mode)
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode=layer_mode,
        )
    return set_pool(
        partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model("LPSLogitLayers"),
            pass_extras=False,
            antialias_layer=antialias_layer,
        ),
        gumbel_softmax=gumbel_softmax,
        softmax=softmax,
        stride=downsampling_factor,
        no_antialias=no_antialias,
        conv=conv,
        projection=projection_layer,
        p_ch=in_channels,
    )


def downsampling_apd(in_channels, downsampling_factor, no_antialias, layer_mode):
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode=layer_mode,
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
        stride=downsampling_factor,
        no_antialias=no_antialias,
    )


def downsampling_lpf(in_channels, downsampling_factor, layer_mode):
    return set_pool(
        partial(
            Decimation,
            antialias_layer=partial(
                LowPassFilter,
                filter_size=3,
                padding="same",
                padding_mode="circular",
                layer_mode=layer_mode,
            ),
        ),
        p_ch=in_channels,
        no_antialias=False,
        stride=downsampling_factor,
    )


def get_upsampling(
    upsampling: Optional[str],
    layer_mode: str = "complex",
    factor: int = 2,
    gumbel_softmax_type: Optional[str] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
) -> nn.Module:
    """Retrieve upsampling layer by name with mode awareness.

    Args:
        upsampling: Upsampling method (e.g., "nearest", "bilinear", "transpose")
        layer_mode: Layer mode to determine real vs complex implementation
        factor: Upsampling factor
        gumbel_softmax_type: Type of Gumbel softmax for LPU variants
        in_channels: Input channels (required for transpose)
        out_channels: Output channels (required for transpose)

    Returns:
        Instantiated upsampling module
    """

    if upsampling is None or upsampling == "none":
        return nn.Identity()

    is_real = is_real_mode(layer_mode)

    if upsampling == "nearest":
        return (
            nn.Upsample(scale_factor=factor, mode="nearest")
            if is_real
            else c_nn.Upsample(scale_factor=factor, mode="nearest")
        )
    elif upsampling == "bilinear":
        return (
            nn.Upsample(scale_factor=factor, mode="bilinear")
            if is_real
            else c_nn.Upsample(scale_factor=factor, mode="bilinear")
        )
    elif upsampling == "bicubic":
        return (
            nn.Upsample(scale_factor=factor, mode="bicubic")
            if is_real
            else c_nn.Upsample(scale_factor=factor, mode="bicubic")
        )
    elif upsampling == "transpose":
        if in_channels is None or out_channels is None:
            raise ValueError(
                "in_channels and out_channels are required for transpose upsampling"
            )
        return (
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=factor,
            )
            if is_real
            else c_nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=factor,
            )
        )
    elif upsampling == "LPU":
        return upsampling_lpu(
            in_channels=in_channels,
            gumbel_softmax_type=gumbel_softmax_type,
            upsampling_factor=factor,
            no_antialias=True,
            layer_mode=layer_mode,
        )
    elif upsampling == "LPU_F":
        return upsampling_lpu(
            in_channels=in_channels,
            gumbel_softmax_type=gumbel_softmax_type,
            upsampling_factor=factor,
            no_antialias=False,
            layer_mode=layer_mode,
        )
    elif upsampling == "APU":
        return upsampling_apu(
            in_channels=in_channels,
            upsampling_factor=factor,
            no_antialias=True,
            layer_mode=layer_mode,
        )
    elif upsampling == "APU_F":
        return upsampling_apu(
            in_channels=in_channels,
            upsampling_factor=factor,
            no_antialias=False,
            layer_mode=layer_mode,
        )
    else:
        raise ValueError(f"Unsupported upsampling method: {upsampling}")


def upsampling_apu(in_channels, upsampling_factor, no_antialias, layer_mode):
    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode=layer_mode,
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


def upsampling_lpu(
    in_channels, gumbel_softmax_type: Optional[str], upsampling_factor, no_antialias, layer_mode
):
    softmax, gumbel_softmax = get_softmax(gumbel_softmax_type, layer_mode=layer_mode)

    antialias_layer = None
    if no_antialias == False:
        antialias_layer = partial(
            LowPassFilter,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode=layer_mode,
        )
    # effective_softmax_type = gumbel_softmax_type
    # if projection is None and gumbel_softmax_type is None:
    #     effective_softmax_type = "mean"

    # softmax, gumbel_softmax = get_softmax(effective_softmax_type, layer_mode=layer_mode)

    return set_unpool(
        partial(
            PolyphaseInvariantUp2D,
            component_selection=LPS_u,
            antialias_layer=antialias_layer,
        ),
        p_ch=in_channels,
        no_antialias=no_antialias,
        softmax=softmax,
        gumbel_softmax=gumbel_softmax,
        stride=upsampling_factor,
    )


def get_projection(
    projection: Optional[str],
    layer_mode: Optional[str] = "complex",
    projection_config: Optional[Dict[str, Any]] = None,
) -> nn.Identity | PolyCtoR | MLPCtoR | c_nn.Mod:
    """Retrieve projection layer by name with mode awareness.

    Args:
        projection: Projection method (e.g., "amplitude", "polynomial", "MLP")
        layer_mode: Layer mode to determine real vs complex implementation
        projection_config: Optional config dict for projection parameters

    Returns:
        Instantiated projection module
    """

    layer_mode = layer_mode or "complex"
    is_real = is_real_mode(layer_mode)
    if is_real or projection is None:
        return nn.Identity()
    elif projection == "polynomial":
        config = projection_config or {}
        order = config.get("order", 3)
        return PolyCtoR(order=order)
    elif projection == "MLP":
        config = projection_config or {}
        hidden_sizes = config.get("hidden_sizes", [8, 16])
        input_size = config.get("input_size", 2)
        output_size = config.get("output_size", 1)
        return MLPCtoR(
            hidden_sizes=hidden_sizes, input_size=input_size, output_size=output_size
        )
    elif projection == "amplitude":
        return c_nn.Mod()
    else:
        raise ValueError(f"Unsupported projection method: {projection}")


def init_weights_mode_aware(module: nn.Module, layer_mode: str) -> None:
    """Initialize weights appropriately based on layer mode.

    Applies mode-appropriate weight initialization strategies:
    - Real modes: Standard PyTorch Kaiming normal initialization
    - Complex modes: Complex-aware Kaiming normal initialization

    Args:
        module (nn.Module): PyTorch module to initialize. Should be a layer
            with weights (Linear, Conv2d, ConvTranspose2d, etc.)
        layer_mode (str): The layer mode (real/complex/split)

    Note:
        Only initializes weights for supported layer types (Linear, Conv2d,
        ConvTranspose2d). Other layer types are ignored.

    Example:
        >>> layer = nn.Linear(10, 5)
        >>> init_weights_mode_aware(layer, "complex")
        >>> # Weights are now initialized for complex-valued operations
    """
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv2d)
        or isinstance(module, nn.ConvTranspose2d)
    ):
        if is_real_mode(layer_mode):
            # Use standard PyTorch initialization for real modes
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        else:
            # Use complex initialization for complex modes
            c_nn.init.complex_kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        # --- modReLU (learnable bias threshold) ---
    elif module.__class__.__name__.lower() == "modrelu":
        # bias value impact the model significantly, 0.25 seems to be a good starting point (empirically)
        if hasattr(module, "b") and module.b is not None:
            with torch.no_grad():
                module.b.fill_(0.25)


def validate_layer_mode(layer_mode: str) -> None:
    """Validate that layer_mode is a supported value.

    Checks that the provided layer_mode is one of the supported modes
    and raises a clear error message if not.

    Args:
        layer_mode (str): The layer mode to validate

    Raises:
        ValueError: If layer_mode is not one of the supported modes

    Example:
        >>> validate_layer_mode("complex")  # No error
        >>> validate_layer_mode("invalid")  # Raises ValueError

    Note:
        This function is automatically called by other mode utilities,
        but can also be used for early validation in user code.
    """
    valid_modes = ["complex", "real", "split"]
    if layer_mode not in valid_modes:
        raise ValueError(
            f"Invalid layer_mode '{layer_mode}'. Must be one of: {', '.join(valid_modes)}"
        )


def get_dropout(
    dropout_prob: float, layer_mode: str, spatial: bool = False
) -> Optional[nn.Module]:
    """Get mode-appropriate dropout layer.

    Returns a dropout layer appropriate for the given layer mode and dropout
    probability. For real modes, uses standard PyTorch dropout. For complex
    modes, uses torchcvnn complex dropout if available, otherwise falls back
    to real dropout with a warning.

    Args:
        dropout_prob (float): Dropout probability. If 0.0, returns None (no dropout).
        layer_mode (str): The layer mode (real/complex/split)
        spatial (bool, optional): Whether to use spatial dropout. Defaults to False.

    Returns:
        Optional[nn.Module]: Dropout layer instance or None if dropout_prob is 0.0

    Raises:
        ValueError: If layer_mode is invalid

    Example:
        >>> # Real mode dropout
        >>> dropout = get_dropout(0.5, layer_mode="real")
        >>> # Complex mode dropout
        >>> dropout = get_dropout(0.3, layer_mode="complex")
        >>> # No dropout
        >>> dropout = get_dropout(0.0, layer_mode="real")  # Returns None

    Note:
        Complex dropout may not be available in all torchcvnn versions.
        In such cases, falls back to real dropout with a warning.
    """
    validate_layer_mode(layer_mode)

    if dropout_prob == 0.0:
        return None

    if dropout_prob < 0.0 or dropout_prob > 1.0:
        raise ValueError(
            f"Dropout probability must be between 0.0 and 1.0, got {dropout_prob}"
        )
    if is_real_mode(layer_mode):
        if spatial:
            return nn.Dropout2d(p=dropout_prob)
        else:
            return nn.Dropout(p=dropout_prob)
    else:
        if spatial:
            return c_nn.Dropout2d(p=dropout_prob)
        else:
            return c_nn.Dropout(p=dropout_prob)