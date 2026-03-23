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