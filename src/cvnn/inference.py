"""
Inference utilities for CVNN.
Handles full-image reconstruction from patches and general model inference.
"""
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from sklearn.mixture import GaussianMixture
import numpy as np

from cvnn.utils import setup_logging
from cvnn.data_processing import dual_real_to_complex_transform

logger = setup_logging(__name__)

def reconstruct_full_image(
    model: torch.nn.Module,
    full_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full image by reassembling patches from `full_loader`.
    Uses mini-batch processing to avoid BatchNorm issues with single samples.

    Args:
        model: The trained model for reconstruction
        full_loader: DataLoader containing image patches
        config: Configuration dictionary containing batch_size
        device: Device to run inference on

    Returns:
        tuple: (original_image, reconstructed_image), each with shape
               (num_channels, nb_rows, nb_cols).
    """
    # determine device, fallback if no parameters
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU device")
            device = torch.device("cpu")
    model.to(device).eval()

    # collect original patches and their indices together
    original_segments: List[np.ndarray] = []
    collected_original_indices: List[int] = []

    for data in full_loader:
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        original_segments.extend([seg for seg in inputs.cpu().numpy()])

        # Extract indices from the same batch
        if len(data) >= 2:
            indices = data[1] if len(data) == 2 else data[2]
            collected_original_indices.extend(indices.cpu().detach().numpy())

    # Verify that segments and indices have the same length
    if len(collected_original_indices) == 0:
        # No indices provided by DataLoader, use sequential placement
        collected_original_indices = None
    elif len(original_segments) != len(collected_original_indices):
        logger.warning(
            f"Mismatch between segments ({len(original_segments)}) and indices ({len(collected_original_indices)}). "
            f"Using sequential placement."
        )
        collected_original_indices = None  # Fall back to sequential placement

    patch_size = config["data"]["dataset"]["patch_size"]

    # Use inferred_input_channels if available, fallback to num_channels for tests
    num_channels = (
        config["data"].get("inferred_input_channels") or config["data"]["num_channels"]
    )
    # Determine actual channels from the data for original image assembly
    actual_channels = (
        original_segments[0].shape[0] if original_segments else num_channels
    )

    original_image = _assemble_image(
        original_segments,
        actual_channels,
        nsamples_per_rows,
        nsamples_per_cols,
        patch_size,
        collected_original_indices,
    )

    # collect reconstructed patches using mini-batch processing
    reconstructed_segments: List[np.ndarray] = []
    collected_reconstructed_indices: List[int] = []

    with torch.no_grad():
        for data in full_loader:
            inputs = data[0] if isinstance(data, (tuple, list)) else data
            outputs = model(inputs.to(device))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]  # Assume first element is the reconstructed output
            reconstructed_segments.extend([seg for seg in outputs.cpu().numpy()])

            # Extract indices from the same batch
            if len(data) >= 2:
                indices = data[1] if len(data) == 2 else data[2]
                collected_reconstructed_indices.extend(indices.cpu().detach().numpy())

    # Determine output channels for reconstructed image
    if reconstructed_segments:
        output_channels = reconstructed_segments[0].shape[0]
    else:
        output_channels = num_channels

        # Verify that segments and indices have the same length
    if collected_original_indices is not None and len(original_segments) != len(
        collected_original_indices
    ):
        logger.warning(
            f"Mismatch between segments ({len(original_segments)}) and indices ({len(collected_original_indices)}). "
            f"Using sequential placement."
        )
        collected_original_indices = None  # Fall back to sequential placement

    # Verify that reconstructed segments and indices have the same length
    if len(collected_reconstructed_indices) == 0:
        # No indices provided by DataLoader, use sequential placement
        collected_reconstructed_indices = None
    elif len(reconstructed_segments) != len(collected_reconstructed_indices):
        logger.warning(
            f"Mismatch between reconstructed segments ({len(reconstructed_segments)}) and indices ({len(collected_reconstructed_indices)}). "
            f"Using sequential placement."
        )
        collected_reconstructed_indices = None  # Fall back to sequential placement

    reconstructed_image = _assemble_image(
        reconstructed_segments,
        output_channels,
        nsamples_per_rows,
        nsamples_per_cols,
        patch_size,
        collected_reconstructed_indices,
    )

    return original_image, reconstructed_image

def reconstruct_full_segmentation(
    model: torch.nn.Module,
    full_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    real_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct full image segmentation by assembling patch predictions.

    Args:
        model: The trained model for segmentation
        full_loader: DataLoader containing image patches
        config: Configuration dictionary
        device: Device to run inference on
        nsamples_per_rows: Number of patch rows
        nsamples_per_cols: Number of patch columns
        real_indices: Optional list of real indices. If None, will be extracted from the dataloader.

    Returns:
        tuple: (original_image, ground_truth, predicted_segmentation)
    """

    # Collect patches, predictions, ground truth, and indices together
    original_patches: List[np.ndarray] = []
    predicted_patches: List[np.ndarray] = []
    ground_truth_patches: List[np.ndarray] = []
    collected_indices: List[int] = []

    with torch.no_grad():
        for batch in full_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)

            # Get predictions
            outputs_non_projected, outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)  # Get class predictions

            # Store patches
            original_patches.extend([patch.cpu().numpy() for patch in inputs])
            predicted_patches.extend([pred.cpu().numpy() for pred in predictions])

            # Extract ground truth labels if available
            if len(batch) >= 2:
                if len(batch) == 2:
                    # batch = [inputs, labels]
                    labels = batch[1]
                    indices = None
                else:
                    # batch = [inputs, labels, indices] or [inputs, indices, labels]
                    if batch[1].dim() > 1:  # labels are usually 2D+ (segmentation maps)
                        labels = batch[1]
                        indices = batch[2] if len(batch) > 2 else None
                    else:  # indices are usually 1D
                        indices = batch[1]
                        labels = batch[2] if len(batch) > 2 else None

                if labels is not None:
                    ground_truth_patches.extend(
                        [label.cpu().numpy() for label in labels]
                    )

                if indices is not None:
                    collected_indices.extend(indices.cpu().detach().numpy())

    # Use collected indices if real_indices not provided
    if real_indices is None:
        real_indices = collected_indices

    # Verify that patches and indices have the same length
    if len(original_patches) != len(real_indices):
        logger.warning(
            f"Mismatch between patches ({len(original_patches)}) and indices ({len(real_indices)}). "
            f"Using sequential placement."
        )
        real_indices = None  # Fall back to sequential placement

    # Get image dimensions from config
    seg_size = config["data"]["dataset"]["patch_size"]

    # Assemble original image
    original_image = _assemble_image(
        original_patches,
        config["data"]["inferred_input_channels"],
        nsamples_per_rows,
        nsamples_per_cols,
        seg_size,
        real_indices,
    )

    # Assemble predicted segmentation
    predicted_segmentation = _assemble_segmentation_image(
        predicted_patches, nsamples_per_rows, nsamples_per_cols, seg_size, real_indices
    )

    # Assemble ground truth segmentation if available
    if ground_truth_patches:
        ground_truth_segmentation = _assemble_segmentation_image(
            ground_truth_patches,
            nsamples_per_rows,
            nsamples_per_cols,
            seg_size,
            real_indices,
        )
    else:
        # Create a dummy ground truth if not available
        logger.warning(
            "No ground truth labels found in dataloader. Creating dummy ground truth."
        )
        ground_truth_segmentation = np.zeros_like(predicted_segmentation)

    return original_image, ground_truth_segmentation, predicted_segmentation


def _assemble_image(
    segments: List[np.ndarray],
    num_channels: int,
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    patch_size: int,
    real_indices: Optional[List[int]] = None,
) -> np.ndarray:
    # Calculate actual image dimensions in pixels
    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    image = np.zeros((num_channels, nb_rows, nb_cols), dtype=segments[0].dtype)

    if real_indices is None:
        # Fall back to sequential placement if no real_indices provided
        idx = 0
        for h in range(0, nb_rows, patch_size):
            for w in range(0, nb_cols, patch_size):
                if (
                    h + patch_size <= nb_rows
                    and w + patch_size <= nb_cols
                    and idx < len(segments)
                ):
                    image[:, h : h + patch_size, w : w + patch_size] = segments[idx]
                    idx += 1
    else:
        # Use real_indices to map segments to their correct positions
        # real_indices[i] tells us the grid position where segments[i] should be placed

        # Place each segment into the correct position
        for segment_index, real_index in enumerate(real_indices):
            if segment_index >= len(segments):
                break

            # Calculate row and col from real_index (sequential patch numbering)
            row = real_index // nsamples_per_cols
            col = real_index % nsamples_per_cols

            # Check bounds
            if row >= nsamples_per_rows or col >= nsamples_per_cols:
                raise ValueError(
                    f"Real index {real_index} maps to position ({row}, {col}) "
                    f"which is out of bounds for grid ({nsamples_per_rows}, {nsamples_per_cols})"
                )

            # Calculate pixel coordinates
            h_start = row * patch_size
            w_start = col * patch_size

            # Ensure we don't exceed image boundaries
            if h_start + patch_size <= nb_rows and w_start + patch_size <= nb_cols:
                image[
                    :, h_start : h_start + patch_size, w_start : w_start + patch_size
                ] = segments[segment_index]

    return image


def _assemble_segmentation_image(
    patches: List[np.ndarray],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    patch_size: int,
    real_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Assemble segmentation patches into full image."""
    # Calculate actual image dimensions
    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    image = np.zeros((nb_rows, nb_cols), dtype=patches[0].dtype)

    if real_indices is None:
        # Fall back to sequential placement if no real_indices provided
        idx = 0
        for h in range(0, nb_rows, patch_size):
            for w in range(0, nb_cols, patch_size):
                if (
                    h + patch_size <= nb_rows
                    and w + patch_size <= nb_cols
                    and idx < len(patches)
                ):
                    image[h : h + patch_size, w : w + patch_size] = patches[idx]
                    idx += 1
    else:
        # Use real_indices to map patches to their correct positions
        for patch_index, real_index in enumerate(real_indices):
            if patch_index >= len(patches):
                break

            # Calculate row and col from real_index (sequential patch numbering)
            row = real_index // nsamples_per_cols
            col = real_index % nsamples_per_cols

            # Check bounds
            if row >= nsamples_per_rows or col >= nsamples_per_cols:
                continue  # Skip out-of-bounds indices

            # Calculate pixel coordinates
            h_start = row * patch_size
            w_start = col * patch_size

            # Ensure we don't exceed image boundaries
            if h_start + patch_size <= nb_rows and w_start + patch_size <= nb_cols:
                image[
                    h_start : h_start + patch_size, w_start : w_start + patch_size
                ] = patches[patch_index]

    return image

def fit_gmm_on_dataloader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
    k_max: int = 30,
    weight_threshold: float = 0.02,
) -> None:
    """
    Extracts latent representations from the dataloader and triggers 
    the Bayesian GMM fitting inside the variational bottleneck.
    """
    # 1. Identifier le bottleneck (Robustesse)
    bn = model.convnet.bottleneck[0]

    model.eval()
    latents_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)
            
            output = model(inputs)
            if isinstance(output, tuple):
                mu = output[1]
            else:
                mu = output 

            latents_list.append(mu.cpu())
            
    # Concatenate all latent tensors (e.g., shape (N_samples, D, H, W) or (N_samples, D))
    latents = torch.cat(latents_list, dim=0)

    if latents.is_complex():
        latents = torch.cat([latents.real, latents.imag], dim=1).numpy()
    else:
        latents = latents.numpy()
    
    # Delegate the actual Bayesian GMM fitting to the bottleneck class
    bn.fit_gmm_prior(latents, k_max=k_max, weight_threshold=weight_threshold)

def inference_on_dataloader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Perform inference on a dataloader and return outputs.
    """
    model.eval()

    # Get a batch of data
    with torch.no_grad():
        batch = next(iter(data_loader))
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dtype == torch.uint8:
                targets = targets.long()
        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
            inputs = batch[0].to(device, non_blocking=True)
            targets = inputs
        else:
            inputs = batch.to(device, non_blocking=True)
            targets = inputs

        outputs = model(inputs)
        if isinstance(outputs, (list, tuple)):
            if len(outputs) >= 4:
                mu = outputs[1]
                std = outputs[2]
                delta = outputs[3]
                outputs = outputs[0]
            elif len(outputs) == 2:
                mu = std = delta = None
                outputs = outputs[1] # assume second element is the projected output 
        else:
            mu = std = delta = None
    return inputs, outputs, targets, mu, std, delta

def get_all_latents(model, data_loader, device, num_samples=2000):
    """
    Extract latent vectors (mu) and labels from the dataloader until num_samples is reached.
    Reuses the robust batch parsing logic from 'inference_on_dataloader'.
    """
    model.eval()
    latents_list = []
    labels_list = []
    total_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, _, targets = batch[0], batch[1], batch[2]
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dtype == torch.uint8:
                    targets = targets.long()
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dtype == torch.uint8:
                    targets = targets.long()
            elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                inputs = batch[0].to(device, non_blocking=True)
                targets = inputs
            else:
                inputs = batch.to(device, non_blocking=True)
                targets = inputs

            # Forward pass
            outputs = model(inputs)
            
            # --- EXTRACTION MU (Logique reprise) ---
            mu = None
            if isinstance(outputs, (list, tuple)):
                if len(outputs) >= 4:
                    mu = outputs[1]
                elif len(outputs) == 2:
                    mu = None 
            else:
                mu = None
            
            if mu is None:
                raise ValueError("The model output structure does not contain 'mu' in the expected position.")

            if mu.ndim > 2:
                mu = torch.flatten(mu, start_dim=1)

            latents_list.append(mu.cpu().numpy())
            if targets.ndim == 1: 
                labels_list.append(targets.cpu().numpy())
            
            total_count += inputs.size(0)
            if total_count >= num_samples:
                break

    latents = np.concatenate(latents_list, axis=0)[:num_samples]
    
    labels = None
    if len(labels_list) > 0:
        labels = np.concatenate(labels_list, axis=0)[:num_samples]
        
    return latents, labels

def sample_from_model_distribution(model: torch.nn.Module, 
                                   num_samples: int = 5,
):
    """
    Generate samples from the model's learned distribution.
    """
    model.eval()

    # Get a batch of data
    with torch.no_grad():
        outputs = model.sample(num_samples)
    return outputs


def _real_dot(u, v):
    if torch.is_complex(u):
        return (u.real * v.real + u.imag * v.imag).sum(dim=-1)
    return (u * v).sum(dim=-1)

def _real_norm(u):
    if torch.is_complex(u):
        return torch.sqrt((u.real**2 + u.imag**2).sum(dim=-1))
    return u.norm(dim=-1)

def slerp(z1, z2, t):
    """
    z1,z2: [..., D] ; t: broadcastable to [..., 1]
    returns: [..., D]
    """
    dot = _real_dot(z1, z2)
    n1, n2 = _real_norm(z1), _real_norm(z2)
    cos_omega = (dot / (n1 * n2 + 1e-12)).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)
    A = torch.sin((1 - t) * omega) / (sin_omega + 1e-12)
    B = torch.sin(t * omega) / (sin_omega + 1e-12)
    out = A.unsqueeze(-1) * z1 + B.unsqueeze(-1) * z2
    # fallback to lerp for tiny angles
    mask = (omega < 1e-3)
    if mask.any():
        out[mask] = ((1 - t[mask].unsqueeze(-1)) * z1[mask] + t[mask].unsqueeze(-1) * z2[mask])
    return out

def lerp(z1, z2, t):
    return (1 - t).unsqueeze(-1) * z1 + t.unsqueeze(-1) * z2

def interpolate_from_dataloader(
    model,
    dataloader,
    num_samples: int = 3,
    mode: str = "slerp",
    device: torch.device = torch.device("cpu"),
):
    """
    Interpolate between 4 samples from the dataloader.
    Handles both Linear (B, D) and Conv (B, D, H, W) latents correctly.
    """
    model.eval()

    # ---- grab 4 images ----
    batch = next(iter(dataloader))
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    assert x.size(0) >= 4, "Need a loader that yields at least 4 samples in a batch."
    x4 = x[:4].to(device)

    # ---- encode → μ ----
    # model(x) returns: recon, mu, std, delta, log_sigma
    _, mu, _, _, _ = model(x4)
    z00, z01, z10, z11 = mu[0], mu[1], mu[2], mu[3]

    D = mu.size(1)
    
    # ---- build grid in latent space ----
    u_vals = torch.linspace(0, 1, num_samples, device=device)
    v_vals = torch.linspace(0, 1, num_samples, device=device)

    # Select interpolation function
    interp_fn = slerp if mode.lower() == "slerp" else lerp

    # Interpolate top and bottom rows
    top_row = torch.stack([interp_fn(z00, z01, t=u) for u in u_vals], dim=0)
    bot_row = torch.stack([interp_fn(z10, z11, t=u) for u in u_vals], dim=0)

    # Interpolate columns
    Z = []
    for j in range(num_samples):
        col = torch.stack([interp_fn(top_row[j], bot_row[j], t=v) for v in v_vals], dim=0)
        Z.append(col)
    Z = torch.stack(Z, dim=1) # [num_samples, num_samples, D]

    # ---- decode the grid ----
    Z_flat = Z.reshape(num_samples * num_samples, D) # [N, D]

    # 1. PREPARE PROBABILITIES (The Patcher Logic)
    list_probs = None
    
    # --- DECODE ---
    # Access bottleneck for projection
    bottleneck_module = model.convnet.bottleneck[0]
    
    # to_input handles the logic (Linear vs Conv) automatically
    z_projected = bottleneck_module.to_input(Z_flat)

    imgs = model.decode(z_projected, probs=list_probs)
    
    # Reshape for visualization
    imgs = imgs.view(num_samples, num_samples, *imgs.shape[1:]).cpu()

    corners = {
        "z00": z00, "z01": z01, "z10": z10, "z11": z11,
        "x00": x4[0], "x01": x4[1], "x10": x4[2], "x11": x4[3],
    }
    return imgs, Z, corners

def sample_from_gmm(model, gmm, num_samples, device):
    """
    Generate samples using the fitted GMM (High Quality Mode).
    """
    # 1. Sample from GMM (Real domain)
    z_gmm_real, _ = gmm.sample(num_samples)
    z_gmm_real = torch.from_numpy(z_gmm_real).float().to(device)
    
    # 2. Convert back to Complex [Re, Im] -> Complex
    dim = z_gmm_real.shape[1] // 2
    z_complex = torch.complex(z_gmm_real[:, :dim], z_gmm_real[:, dim:])
    
    # 3. Decode
    model.eval()
    with torch.no_grad():
        # Attention: il faut projeter z si tu utilises un bottleneck variatinonnel
        # model.convnet.bottleneck[0] est le VariationalBottleneck
        # Si ton modèle attend 'z' direct dans decode, c'est bon.
        # Check si bottleneck a une méthode 'to_input' ou si decode gère.
        # Dans ton code actuel: decode prend z. 
        recons = model.decode(z_complex)
        
    return recons

def perform_phase_twist(
    model: torch.nn.Module,
    batch: torch.Tensor,
    steps: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Performs a 'Phase Twist' trajectory in latent space.
    z_new = z * exp(i * theta) for theta in [0, 2pi].
    
    Returns:
        Tensor of shape [steps, C, H, W] containing the decoded frames.
    """
    model.eval()
            
    with torch.no_grad():
        # 1. Encode to get latent center (mu)
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        mu = outputs[1] # mu is the 2nd element

        input_sample = inputs[0:1] if inputs.shape[0] > 1 else inputs # Take the first sample for the trajectory
        mu = mu[0:1] if mu.shape[0] > 1 else mu # Take the first sample for the trajectory

        # 2. Generate trajectory
        thetas = torch.linspace(0, 2 * np.pi, steps, device=device)
        decoded_frames = []

        for theta in thetas:
            # Create rotation factor e^(i*theta)
            # Note: We apply global phase shift to the latent code
            
            if torch.is_complex(mu):
                # --- Complex Case (C-VAE) ---
                # Natural phase rotation
                rotator = torch.polar(torch.ones_like(mu.real), torch.full_like(mu.real, theta))
                z_rotated = mu * rotator
            else:
                # --- Real Case (Baseline) ---
                # We force an interpretation of Real latents as pseudo-complex pairs to show it fails.
                # Shape [B, D] -> View as [B, D/2, 2] -> Rotate -> Flatten back
                B, D = mu.shape
                if D % 2 != 0:
                     # If odd dimension, just return original (can't rotate pairs)
                     z_rotated = mu 
                else:
                    mu_pairs = mu.view(B, D // 2, 2)
                    # Rotation matrix for 2D vectors
                    # | cos -sin |
                    # | sin  cos |
                    cos_t = torch.cos(theta)
                    sin_t = torch.sin(theta)
                    
                    # Manual rotation
                    # x' = x cos - y sin
                    # y' = x sin + y cos
                    x = mu_pairs[..., 0]
                    y = mu_pairs[..., 1]
                    x_new = x * cos_t - y * sin_t
                    y_new = x * sin_t + y * cos_t
                    
                    z_rotated = torch.stack([x_new, y_new], dim=-1).view(B, D)

            # 3. Decode            
            # Manually project and decode
            bottleneck = model.convnet.bottleneck[0]
            z_proj = bottleneck.to_input(z_rotated)
            recon = model.decode(z_proj) # Or model.decode(z_proj)

            decoded_frames.append(recon.cpu())

    return input_sample, torch.cat(decoded_frames, dim=0) # [steps, C, H, W]