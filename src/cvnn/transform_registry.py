"""
Transform registry for automatic transform selection based on configuration.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, Type

# Third-party imports
import torch
import torchcvnn.transforms as c_transforms
from torchvision import transforms
import numpy as np

# Local imports
from cvnn.utils import setup_logging

logger = setup_logging(__name__)

class ClampTensor(object):
    """
    Clamp tensor values between min_val and max_val.
    """
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor, eps=1e-10):
        if tensor.is_complex():
            amplitude = torch.abs(tensor)
            clipped_amplitude = torch.clamp(amplitude, self.min_val, self.max_val)
            scale = clipped_amplitude / (amplitude + eps)
            return tensor * scale
        else:
            return torch.clamp(tensor, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + f'(min={self.min_val}, max={self.max_val})'

class GlobalScalarNormalize(object):
    """
    Applies a global scalar standardization to a complex tensor.
    X_norm = (X - global_mean) / global_std
    This strictly preserves the Degree of Impropriety (rho) of the dataset.
    Calculates the global stats directly from complex means and 2x2 covariance matrices.
    """
    def __init__(self, means, covs, eps=1e-10):
        self.eps = eps
        
        # Convert inputs to numpy arrays for easy manipulation
        means_np = np.array(means, dtype=np.float32)
        covs_np = np.array(covs, dtype=np.float32)
        
        # Handle single channel (1D means, 2D covs) vs multi-channel
        if means_np.ndim == 1:
            means_np = means_np.reshape(1, 2)
        if covs_np.ndim == 2:
            covs_np = covs_np.reshape(1, 2, 2)
            
        # 1. Compute complex means: Re + j*Im
        complex_means = means_np[:, 0] + 1j * means_np[:, 1]
        self.means = torch.tensor(complex_means, dtype=torch.complex64).view(-1, 1, 1)
        
        # 2. Compute global scalar standard deviation: sqrt(Trace) = sqrt(V_rr + V_ii)
        traces = covs_np[:, 0, 0] + covs_np[:, 1, 1]
        self.stds = torch.tensor(np.sqrt(traces), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        if not tensor.is_complex():
            tensor = tensor.to(torch.complex64)
            
        # Move params to the same device as tensor
        means = self.means.to(tensor.device)
        stds = self.stds.to(tensor.device)
        
        # Apply standardisation
        return (tensor - means) / (stds + self.eps)

    def __repr__(self):
        mean_str = [f"{m.item():.4f}" for m in self.means.flatten()]
        std_str = [f"{s.item():.4f}" for s in self.stds.flatten()]
        return self.__class__.__name__ + f'(derived_means={mean_str}, derived_stds={std_str})'
    
class TransposeNumpy(object):
    """
    Transpose numpy array from (H, W, C) to (C, H, W).
    """
    def __call__(self, img):
        if isinstance(img, np.ndarray) and img.ndim == 3:
            return img.transpose(2, 0, 1)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
# Available transforms by layer mode
_TRANSFORM_REGISTRY: Dict[str, Dict[str, Type]] = {
    "real": {
        "resize": transforms.Resize,
        "normalize": transforms.Normalize,
        "randomhorizontalflip": transforms.RandomHorizontalFlip,
        "randomverticalflip": transforms.RandomVerticalFlip,
        "randomrotation": transforms.RandomRotation,
        "randomcrop": transforms.RandomCrop,
        "colorjitter": transforms.ColorJitter,
        "centercrop": transforms.CenterCrop,
        "pad": transforms.Pad,
        "fft2": c_transforms.FFT2,
        "logamplitude": c_transforms.LogAmplitude,
        "clamp": ClampTensor,
    },
    "complex": {
        "fftresize": c_transforms.FFTResize,
        "spatialresize": c_transforms.SpatialResize,
        "centercrop": c_transforms.CenterCrop,
        "padifneeded": c_transforms.PadIfNeeded,
        "randomphase": c_transforms.RandomPhase,
        "randomhorizontalflip": transforms.RandomHorizontalFlip,
        "randomverticalflip": transforms.RandomVerticalFlip,
        "randomrotation": transforms.RandomRotation,
        "randomcrop": transforms.RandomCrop,
        "global_scalar_normalize": GlobalScalarNormalize,
        "normalize": c_transforms.Normalize,
        "logamplitude": c_transforms.LogAmplitude,
        "fft2": c_transforms.FFT2,
        "clamp": ClampTensor,
    },
}


def register_transform(
    layer_mode: str, transform_name: str, transform_class: Type
) -> None:
    """
    Register a transform for a specific layer mode.

    Args:
        layer_mode: Layer mode ("real" or "complex")
        transform_name: Name of the transform (e.g., "resize", "normalize")
        transform_class: Transform class to register
    """
    if layer_mode not in _TRANSFORM_REGISTRY:
        _TRANSFORM_REGISTRY[layer_mode] = {}

    _TRANSFORM_REGISTRY[layer_mode][transform_name] = transform_class
    logger.debug(f"Registered transform: {layer_mode}.{transform_name}")


def get_transform_class(layer_mode: str, transform_name: str, dataset_type: Optional[str] = None, is_fft: bool = False) -> Type:
    """
    Get transform class for a specific layer mode.

    Args:
        layer_mode: Layer mode ("real" or "complex")
        transform_name: Name of transform (e.g., "resize")
        dataset_type: Type of dataset (e.g., "polsar", "sar", "grayscale"), optional

    Returns:
        Transform class

    Raises:
        ValueError: If layer mode or transform name is unknown
    """
    # Determine which registry to use based on layer mode
    if layer_mode == "complex" or is_fft or dataset_type in ["polsar", "sar", "mri"]:
        registry_key = "complex"
    elif layer_mode == "real" or layer_mode == "split":
        registry_key = "real"
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")

    if registry_key not in _TRANSFORM_REGISTRY:
        raise ValueError(f"No transforms registered for layer_mode: {layer_mode}")

    transforms_for_mode = _TRANSFORM_REGISTRY[registry_key]
    if transform_name not in transforms_for_mode:
        available = list(transforms_for_mode.keys())
        raise ValueError(
            f"Transform '{transform_name}' not available for layer_mode '{layer_mode}'. "
            f"Available: {available}"
        )

    return transforms_for_mode[transform_name]


def _get_dataset_base_transforms(
    dataset_type: str, num_channels: int, dataset_name: Optional[str] = None
    ) -> List:
    """Get dataset-specific base transforms based on dataset type."""
    base_transforms = []
    
    if dataset_name == "MSTARTargets":
        base_transforms.append(TransposeNumpy())

    if dataset_type in ["sar", "polsar"]:
        # SAR/PolSAR datasets need PolSAR and LogAmplitude
        base_transforms.append(
            c_transforms.PolSAR(out_channel=num_channels)
        )
        
    return base_transforms


def _get_mode_conversion_transforms(
    layer_mode: str, real_pipeline_type: Optional[str] = None, dataset_type: Optional[str] = None, is_fft: bool = False, for_stats: bool = False
) -> List:
    """Get mode-specific conversion transforms."""
    if for_stats:
        if dataset_type in ["polsar", "sar", "mri"]:
            return [c_transforms.ToTensor(dtype="complex64")]
        else:
            return [transforms.ToTensor()]
    else:
        if layer_mode in ["complex", "split"]:
            if dataset_type in ["polsar", "sar", "mri"]: # do not include fft transformed data
                # Complex modes use complex64 tensor
                return [c_transforms.ToTensor(dtype="complex64")]
            else:
                return [transforms.ToTensor()] # for fft transformed data, use real tensor
        else:
            if real_pipeline_type == "real_real":
                return [transforms.ToTensor()]
            elif real_pipeline_type == "complex_amplitude_real":
                if dataset_type in ["polsar", "sar", "mri"]:
                    return [
                        c_transforms.ToTensor(dtype="complex64"),
                        c_transforms.Amplitude(dtype="float32"),
                    ]
                else:
                    return [
                        transforms.ToTensor(),
                        c_transforms.Amplitude(dtype="float32"),
                    ]
            elif real_pipeline_type == "complex_dual_real":
                if dataset_type in ["polsar", "sar", "mri"]:
                    return [
                        c_transforms.ToTensor(dtype="complex64"),
                        c_transforms.RealImaginary(dtype="float32"),
                    ]
                else:
                    return [
                        transforms.ToTensor(),
                        c_transforms.RealImaginary(dtype="float32"),
                    ]
            else:
                raise ValueError(f"Unknown real_pipeline_type: {real_pipeline_type}")

def build_transform_pipeline(cfg: Dict[str, Any], for_stats: bool = False, is_train: bool = True) -> transforms.Compose:
    """
    Build complete transform pipeline based on configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        Composed transform pipeline
    """
    # Extract configuration
    layer_mode = cfg.get("model").get("layer_mode")
    real_pipeline_type = cfg.get("data").get("real_pipeline_type")
    num_channels = cfg["data"].get("num_channels")
    is_fft = True if any(t.get("name") == "fft2" for t in cfg["data"].get("transforms", [])) else False 

    mean_real_value = cfg["data"].get("mean_real_value") #mean for real pipeline
    std_real_value = cfg["data"].get("std_real_value") #std for real pipeline

    min_real_value = cfg["data"].get("min_real_value") 
    max_real_value = cfg["data"].get("max_real_value")

    mean_complex_value = cfg["data"].get("mean_complex_value") #means for complex pipeline
    cov_complex_value = cfg["data"].get("cov_complex_value") #covariances for complex pipeline

    dataset_type = cfg["data"].get("type", "unknown")
    dataset_name = cfg.get("data", {}).get("dataset", {}).get("name")

    # Start with dataset-specific base transforms
    transform_list = _get_dataset_base_transforms(
        dataset_type, num_channels, dataset_name
    )

    # Add mode-specific conversion transforms
    mode_transforms = _get_mode_conversion_transforms(layer_mode, real_pipeline_type, dataset_type, is_fft,for_stats)
    transform_list.extend(mode_transforms)

    AUGMENTATION_TRANSFORMS = {
        "randomhorizontalflip", "randomverticalflip", "randomrotation", 
        "randomcrop", "colorjitter", "randomphase"
    }
    
    # Add configurable transforms from config
    additional_transforms = cfg.get("data", {}).get("transforms", [])
    for transform_config in additional_transforms:
        transform_name = transform_config["name"].lower()

        if not is_train and transform_name in AUGMENTATION_TRANSFORMS:
            logger.debug(f"Skipping augmentation {transform_name} for validation/test.")
            continue
        
        transform_params = transform_config.get("params", {})

        # Convert size parameter to tuple if it's a list (for transforms that expect tuples)
        if "size" in transform_params and isinstance(transform_params["size"], list):
            transform_params = transform_params.copy()  # Don't modify original
            transform_params["size"] = tuple(transform_params["size"])

        if transform_name == "normalize":
            if layer_mode == "complex" or real_pipeline_type in ["complex_amplitude_real", "complex_dual_real"]:
                if mean_complex_value is not None and cov_complex_value is not None:
                    transform_params = transform_params.copy()  # Don't modify original
                    transform_params["means"] = mean_complex_value
                    transform_params["covs"] = cov_complex_value
                else:
                    raise ValueError(
                        "Normalization requires either mean/cov or min/max values in config."
                    )
            elif layer_mode in ["real", "split"]:
                if (layer_mode == "real" and real_pipeline_type in ["real_real"]):
                    if mean_real_value is not None and std_real_value is not None:
                        transform_params = transform_params.copy()  # Don't modify original
                        transform_params["mean"] = mean_real_value
                        transform_params["std"] = std_real_value
                    else:
                        raise ValueError(
                            "Normalization requires mean and std values in config."
                        )
        elif transform_name == "global_scalar_normalize":
            if mean_complex_value is not None and cov_complex_value is not None:
                transform_params = transform_params.copy()
                transform_params["means"] = mean_complex_value
                transform_params["covs"] = cov_complex_value
            else:
                raise ValueError(
                    "GlobalScalarNormalize requires mean_complex_value and cov_complex_value in config."
                )
        elif transform_name == "logamplitude":
            if min_real_value is not None and max_real_value is not None:
                transform_params = transform_params.copy()  # Don't modify original
                transform_params["min_value"] = np.min(min_real_value)
                transform_params["max_value"] = np.max(max_real_value)
            else:
                raise ValueError(
                    "LogAmplitude transform requires min_value and max_value in config."
                )
        try:
            transform_class = get_transform_class(layer_mode, transform_name, dataset_type, is_fft)
            transform_instance = transform_class(**transform_params)
            transform_list.append(transform_instance)
            logger.debug(
                f"Added transform: {transform_name} with params: {transform_params}"
            )
        except Exception as e:
            logger.warning(f"Failed to add transform {transform_name}: {e}")
    
    if not for_stats and real_pipeline_type in ["complex_dual_real", "complex_amplitude_real"]:
        # Find and move RealImaginary or Amplitude transform to the end
        for i, transform in enumerate(transform_list):
            if isinstance(transform, (c_transforms.RealImaginary, c_transforms.Amplitude)):
                transform_list.append(transform_list.pop(i))
                break
    
    return transforms.Compose(transform_list)
