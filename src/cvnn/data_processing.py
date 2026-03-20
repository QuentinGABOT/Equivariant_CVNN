# Standard library imports
from typing import Any, Dict, Union, Tuple, Optional


# Third-party imports
import numpy as np
import torch
from skimage import exposure

# Local imports
from cvnn.utils import (
    safe_log, 
    setup_logging,
)

logger = setup_logging(__name__)

def dual_real_to_complex_transform(tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert a tensor with real and imaginary parts in separate channels to a complex tensor.
    This transformation is necessary when the model expects complex inputs but the data is provided in a dual-channel format.

    Supports multiple channel formats:
    - Single channel: (B, 2, H, W) or (2, H, W) where dim contains [real, imag]
    - Multi-channel: (B, 2C, H, W) or (2C, H, W) where first C channels are real, last C are imaginary

    Args:
        tensor: Input tensor with shape (B, 2, H, W), (2, H, W), (B, 2C, H, W), or (2C, H, W).

    Returns:
        Complex tensor with shape (B, 1, H, W), (1, H, W), (B, C, H, W), or (C, H, W) respectively.

    Raises:
        ValueError: If input tensor does not have expected number of dimensions or even number of channels.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if tensor.ndim == 4:
        # Case: (B, C, H, W) where C must be even
        num_channels = tensor.shape[1]
        if num_channels % 2 != 0:
            raise ValueError(f"Expected even number of channels, got {num_channels}")
        C = num_channels // 2
        real_part = tensor[:, :C, :, :]
        imag_part = tensor[:, C:, :, :]
        return torch.complex(real_part, imag_part)
    elif tensor.ndim == 3:
        # Case: (C, H, W) where C must be even
        num_channels = tensor.shape[0]
        if num_channels % 2 != 0:
            raise ValueError(f"Expected even number of channels, got {num_channels}")
        C = num_channels // 2
        real_part = tensor[:C, :, :]
        imag_part = tensor[C:, :, :]
        return torch.complex(real_part, imag_part)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
    
def inverse_logamplitude(
    tensor: Union[np.ndarray, torch.Tensor], max_value: float, min_value: float
) -> torch.Tensor:
    """
    Apply exponential amplitude transformation with specified bounds.

    This transformation maps normalized amplitudes [0,1] to a physically meaningful
    range [m, M] in dB scale for better contrast.

    Args:
        tensor: Input complex SAR tensor (numpy array or torch tensor)
        max_value: Maximum amplitude value in dB (e.g., 30 dB) if None, no upper clipping is applied
        min_value: Minimum amplitude value in dB (e.g., -30 dB) if None, no lower clipping is applied

    Returns:
        Transformed complex tensor with enhanced amplitude range

    Raises:
        ValueError: If input tensor is not complex-valued
    """
    # Convert to torch tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if not torch.is_complex(tensor):
        tensor = dual_real_to_complex_transform(tensor)

    amplitude = torch.abs(tensor)
    phase = torch.angle(tensor)

    # More robust log transformation
    log_M = safe_log(max_value, base=10)
    log_m = safe_log(min_value, base=10)

    # Enhanced amplitude transformation with better numerical stability
    inv_transformed_amplitude = torch.clamp(
        torch.exp(((log_M - log_m) * amplitude + log_m) * np.log(10)),
        min=0.0,
        max=1e9,  # Prevent overflow
    )

    # Recombine amplitude and phase
    new_tensor = inv_transformed_amplitude * torch.exp(1j * phase)
    return new_tensor

def denormalize_global_scalar_complex(tensor: torch.Tensor, means, covs, eps: float=1e-10) -> torch.Tensor:
    """
    Exact inverse of GlobalScalarNormalize.
    X = X_norm * global_std + complex_mean
    """
    means_np = np.array(means, dtype=np.float32)
    covs_np = np.array(covs, dtype=np.float32)
    
    if means_np.ndim == 1:
        means_np = means_np.reshape(1, 2)
    if covs_np.ndim == 2:
        covs_np = covs_np.reshape(1, 2, 2)
        
    complex_means = means_np[:, 0] + 1j * means_np[:, 1]
    complex_means = torch.tensor(complex_means, dtype=torch.complex64).to(tensor.device).view(-1, 1, 1)
    
    traces = covs_np[:, 0, 0] + covs_np[:, 1, 1]
    stds = torch.tensor(np.sqrt(traces), dtype=torch.float32).to(tensor.device).view(-1, 1, 1)
    return tensor * (stds + eps) + complex_means

def denormalize_complex_transform(
    tensor: Union[np.ndarray, torch.Tensor], 
    means, 
    covs, 
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Apply denormalization transform to the input tensor.
    This transformation denormalizes the input tensor using the provided means and covariance matrices.
    
    Agnostic to input shape, provided the channel dimension follows standard conventions:
    - 3D Input: [C, H, W] -> Channel is dim 0
    - 4D+ Input: [B, C, H, W, ...] -> Channel is dim 1

    Args:
        tensor: Input complex tensor (numpy array or torch tensor)
        means: Mean vector for normalization
        covs: Covariance matrix for normalization
        eps: Small value to ensure numerical stability

    Returns:
        Normalized complex tensor

    Raises:
        ValueError: If input tensor is not complex-valued.
    """
    # 1. Convert to torch tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if not tensor.dtype.is_complex:
        raise ValueError(f"Expected complex tensor, got {tensor.dtype}")

    # 2. Compute transformation matrix A on CPU using NumPy
    # (Kept identical to original logic to ensure mathematical consistency)
    means = np.asarray(means, dtype=np.float64)
    covs  = np.asarray(covs,  dtype=np.float64)
    covs  = 0.5 * (covs + covs.transpose(0, 2, 1))
    w, V  = np.linalg.eigh(covs)
    w     = np.maximum(w, eps)
    D     = np.zeros_like(covs)
    D[:, 0, 0] = np.sqrt(w[:, 0])
    D[:, 1, 1] = np.sqrt(w[:, 1])
    A_np  = V @ D @ np.transpose(V, (0, 2, 1))  # Shape (C, 2, 2)

    # 3. Move components to device
    # A shape: [C, 2_out, 2_in]
    A  = torch.as_tensor(A_np, device=tensor.device, dtype=tensor.real.dtype)
    mu = torch.as_tensor(means, device=tensor.device, dtype=tensor.real.dtype)

    # 4. Handle Dimensions Agnostically
    # Determine which dimension is the Channel dimension (C)
    # Convention: dim 0 for unbatched (3D), dim 1 for batched (4D+)
    c_dim = 0 if tensor.ndim == 3 else 1
    C = A.shape[0]

    if tensor.shape[c_dim] != C:
        raise ValueError(f"Tensor channel dimension ({tensor.shape[c_dim]}) "
                         f"does not match mean/covs size ({C})")

    # View as real: Shape becomes [..., C, ..., 2] (Real/Imag adds a dim at the end)
    # We use view_as_real to avoid copying data immediately
    X_real = torch.view_as_real(tensor)
    
    # Move Channel dimension to 0 to standardize processing
    # New Shape: [C, ..., 2]
    X_permuted = X_real.movedim(c_dim, 0)

    # 5. Apply Transformation
    # A: [C, out_comp, in_comp]
    # X_permuted: [C, ..., in_comp]
    # Einsum 'cij, c...j -> c...i' explanation:
    #   c:   Channel dimension (matched)
    #   ...: Any number of batch/spatial dimensions
    #   j:   Input component (real/imag)
    #   i:   Output component (real/imag)
    X_transformed = torch.einsum('cij, c...j -> c...i', A, X_permuted)

    # Add Mean
    # mu is [C, 2]. We need to broadcast it to [C, 1, ..., 1, 2]
    # We create a view with singleton dimensions for all spatial dims
    shape_broadcast = [C] + [1] * (X_transformed.ndim - 2) + [2]
    X_transformed = X_transformed + mu.view(*shape_broadcast)

    # 6. Restore Dimensions
    # Move C back to its original position
    X_out = X_transformed.movedim(0, c_dim)
    
    # Return as complex tensor
    # Ensure contiguity before viewing as complex to avoid stride errors
    return torch.view_as_complex(X_out.contiguous())
    
def denormalize_real_transform(
    tensor: Union[np.ndarray, torch.Tensor], mean, std, eps: float = 1e-10
) -> torch.Tensor:
    """
    Apply denormalization transform to the input tensor.
    This transformation denormalizes the input tensor using the provided mean and standard deviation.

    Args:
        tensor: Input real tensor (numpy array or torch tensor)
        mean: Mean value for normalization
        std: Standard deviation for normalization
        eps: Small value to ensure numerical stability

    Returns:
        Normalized real tensor

    Raises:
        ValueError: If input tensor is not real-valued 
    """
    # Convert to torch tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if tensor.dtype.is_complex:
        raise ValueError(f"Expected real tensor, got {tensor.dtype}")

    mean = np.asarray(mean, dtype=np.float64)
    std  = np.asarray(std,  dtype=np.float64)
    std  = np.maximum(std, eps)

    mean = torch.as_tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std  = torch.as_tensor(std,  device=tensor.device, dtype=tensor.dtype)

    return (tensor - mean.view(-1,1,1)) / std.view(-1,1,1)

def equalize(
    image: np.ndarray, p1: Optional[float] = None, p2: Optional[float] = None, percentiles: Optional[Tuple[float, float]] = (5,95)
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Automatically adjust contrast of the SAR image using histogram equalization.

    Args:
        image: Input SAR image (intensity or amplitude)
        p1: Lower percentile for clipping (if None, computed automatically)
        p2: Upper percentile for clipping (if None, computed automatically)
        percentiles: Optional tuple of the percentiles to compute if p1 or p2 are not provided (default is (5, 95))

    Returns:
        Tuple of (equalized_image, (p1_used, p2_used))
    Raises:
        ValueError: If input image is empty
    """
    if image.size == 0:
        raise ValueError("Input image cannot be empty")

    # Convert to log scale for better visualization
    img_log = 20*safe_log(np.abs(image), base=10)

    # Compute percentiles if not provided
    if p1 is None or p2 is None:
        p1_computed, p2_computed = np.percentile(img_log, percentiles)
        p1_final = p1 if p1 is not None else p1_computed
        p2_final = p2 if p2 is not None else p2_computed
    else:
        p1_final, p2_final = p1, p2

    # Ensure valid range
    if p1_final >= p2_final:
        logger.warning(
            f"Invalid percentile range: p1={p1_final}, p2={p2_final}. Using image min/max."
        )
        p1_final, p2_final = float(np.min(img_log)), float(np.max(img_log))
    # Rescale intensity (type checker issue with scikit-image)
    img_rescaled = exposure.rescale_intensity(
        img_log,
        in_range=(p1_final, p2_final),  # type: ignore
        out_range=(0, 1),  # type: ignore
    )

    # Convert to uint8
    img_final = np.round(img_rescaled * 255).astype(np.uint8)

    return img_final, (p1_final, p2_final)

def revert_transforms(data: Union[np.ndarray, torch.Tensor], cfg: Dict[str, Any], eps: float=1e-10) -> np.ndarray:
    """
    Inverse logical data transformations based on configuration.
    """
    # 1. Manage types (Tensor -> Numpy)
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach()
    
    # 2. Manage dimensions (ensure C, H, W or B, C, H, W)
    # If shape is (C, H, W) assume single sample and add batch dim
    assert data.ndim in [3,4], f"Expected 3D or 4D data, got shape {data.shape}"

    # 3. Retrieve config
    layer_mode = cfg.get("model", {}).get("layer_mode", "unknown").lower()
    is_complex = layer_mode in ["complex", "split"]
    is_fft = True if any(t.get("name") == "fft2" for t in cfg["data"].get("transforms", [])) else False 
    is_dual_real = cfg.get("data").get("real_pipeline_type", None)
    transforms_cfg = cfg.get("data", {}).get("transforms", [])
    transform_names = [t["name"].lower() for t in transforms_cfg]

    if is_dual_real is not None:
        if is_dual_real == "complex_dual_real":
            data = dual_real_to_complex_transform(data)
            is_complex = True
    mean_value = cfg["data"].get("mean_real_value")
    if mean_value is not None:
        mean_value = np.max(mean_value)
    std_value = cfg["data"].get("std_real_value")
    if std_value is not None:
        std_value = np.max(std_value)
    max_value = cfg["data"].get("max_real_value")
    if max_value is not None:
        max_value = np.max(max_value)
    min_value = cfg["data"].get("min_real_value")
    if min_value is not None:
        min_value = np.min(min_value)

    if is_complex or is_fft:
        means = cfg["data"].get("mean_complex_value")
        covs = cfg["data"].get("cov_complex_value")
    else:
        means = cfg["data"].get("mean_real_value")
        covs = cfg["data"].get("std_real_value")
    
    # 4. Logical inverse transforms
    if "logamplitude" in transform_names:
        data = inverse_logamplitude(data, max_value, min_value)
    else:
        if "global_scalar_normalize" in transform_names or "normalize" in transform_names:
            if is_complex or is_fft:
                if is_fft:
                    # In FFT mode with real model, data is real but was normalized as complex
                    # So we need to convert to complex first
                    if (len(data.shape) == 4 and data.shape[1] == 2):
                        # Cas Standard (B, 2, H, W) -> On sépare et on combine
                        data = data.unsqueeze(0)
                        data = data.permute(1, 0, 3, 4, 2).contiguous() # (B, 1, H, W, 2)
                        data = torch.view_as_complex(data)
                    elif len(data.shape) == 3 and data.shape[0] == 2:
                        # Cas Standard (2, H, W) -> On sépare et on combine
                        data = data.unsqueeze(0).unsqueeze(0)
                        data = data.permute(1, 0, 3, 4, 2).contiguous() # (1, 1, H, W, 2)
                        data = torch.view_as_complex(data)
                if "global_scalar_normalize" in transform_names:
                    data = denormalize_global_scalar_complex(data, means, covs, eps)
                else:
                    data = denormalize_complex_transform(data, means, covs, eps)        
            else:
                data = denormalize_real_transform(data, means, covs, eps)
    
    if "fft2" in transform_names:
        data = torch.fft.ifftshift(data, dim=(-2, -1))
        data = torch.fft.ifft2(data).real

    return data.numpy()