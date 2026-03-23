# Standard library imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from threadpoolctl import threadpool_limits

# Local imports
from cvnn.utils import setup_logging
from cvnn.physics import (
    pauli_transform, 
    krogager_transform,
    h_alpha,
    compute_h_alpha_coords,
    draw_halpha_zones,
    compute_physical_boundary,
    cameron,
)
from cvnn.data_processing import (
    equalize,
)

logger = setup_logging(__name__)

def _display_img(img: np.ndarray, dataset_type: str, percentiles: Optional[Tuple[float, float]] = None) -> None:
    p1, p2 = None, None
    if dataset_type == "polsar":
        img = pauli_transform(img)
        if percentiles is None:
            img, (p1, p2) = equalize(img)
        else:
            p1, p2 = percentiles
            img, _ = equalize(img, p1=p1, p2=p2)
        img = img.transpose(1, 2, 0)
        cmap = None
        origin = "lower"
    elif dataset_type == "sar":
        if percentiles is None:
            img, (p1, p2) = equalize(img)
        else:
            p1, p2 = percentiles
            img, _ = equalize(img, p1=p1, p2=p2)
        img = img.squeeze()
        cmap = "gray"
        origin = "lower"
    elif dataset_type == "grayscale":
        img = img.squeeze()
        cmap = "gray"
        origin = "upper"
    elif dataset_type == "mri":
        if percentiles is None:
            img, (p1, p2) = equalize(img)
        else:
            p1, p2 = percentiles
            img, _ = equalize(img, p1=p1, p2=p2)
        img = img.squeeze()
        cmap = "gray"
        origin = "lower"
    elif dataset_type == "rgb":
        img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        cmap = None
        origin = "upper"
    else:
        img = img.transpose(1, 2, 0)
        cmap = "gray" if img.shape[2] == 1 else None
        origin = "upper"

    return img, cmap, origin, (p1, p2)

def _get_phase_img(img_np, dataset_type: str) -> np.ndarray:
    if dataset_type == "polsar":
        phase_img = np.angle(img_np[0])
    elif dataset_type == "mri":
        phase_img = np.angle(img_np[0]) if img_np.ndim > 2 else np.angle(img_np)
    else:
        phase_img = np.angle(img_np)
    return phase_img.squeeze()

def plot_phase(image: np.ndarray) -> np.ndarray:
    """
    Convert phase information to displayable format normalized to [0, 255].

    Args:
        image: Complex-valued SAR image array

    Returns:
        Phase image normalized to uint8 range [0, 255]

    Raises:
        ValueError: If input is not complex-valued
    """
    if not np.iscomplexobj(image):
        raise ValueError("Input image must be complex-valued")

    phase_image = np.angle(image)  # Phase in [-π, π)
    # Normalize phase to [0, 1]
    normalized_phase = (phase_image + np.pi) / (2 * np.pi)
    # Scale to [0, 255] and convert to integer
    scaled_phase = np.round(normalized_phase * 255).astype(np.uint8)
    return scaled_phase

def plot_losses(
    train_losses: Sequence[float],
    valid_losses: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """Plot train vs. validation loss over epochs.
    Args:
        train_losses: Sequence of training loss values
        valid_losses: Optional sequence of validation loss values
    Returns:
        Matplotlib figure object containing the loss curves.
    """
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    if valid_losses is not None:
        plt.plot(epochs, valid_losses, label="Validation")
    return fig

def plot_reconstructions(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    dataset_type: str,
    num_samples: int = 5,
    show_spectrum: bool = False
) -> plt.Figure:
    """
    Show original and reconstructed images.
    If show_spectrum is True, adds one row per channel displaying the Fourier Amplitude.

    Args:
        inputs: (B, C, H, W)
        outputs: (B, C, H, W)
        dataset_type: "polsar", "sar", "grayscale", etc.
        num_samples: Number of samples to display
        show_spectrum: If True, adds C rows (one per channel) for frequency analysis.
    """
    B, C, H, W = inputs.shape
    num_samples = min(num_samples, B)
    
    nrows = 1 + C if show_spectrum else 1
    num_cols = num_samples * 2
    
    figsize_h = 3.5 * nrows 
    figsize_w = 3.5 * num_samples
    
    fig, axes = plt.subplots(nrows, num_cols, figsize=(figsize_w, figsize_h), 
                             squeeze=False, constrained_layout=True)

    for i in range(num_samples):
        col_orig = i * 2
        col_recon = i * 2 + 1
        
        img_i = inputs[i]
        img_o = outputs[i]
        
        vis_i, cmap, origin, (p1, p2) = _display_img(img_i, dataset_type)
        vis_o, _, _, _ = _display_img(img_o, dataset_type)

        ax_s_orig = axes[0][col_orig]
        ax_s_orig.imshow(vis_i, cmap=cmap, origin=origin)
        if i == 0: ax_s_orig.set_ylabel("Spatial Domain", fontsize=10, fontweight='bold')
        ax_s_orig.set_title("Original" if i == 0 else "")
        ax_s_orig.set_xticks([])
        ax_s_orig.set_yticks([])

        ax_s_recon = axes[0][col_recon]
        ax_s_recon.imshow(vis_o, cmap=cmap, origin=origin)
        ax_s_recon.set_title("Reconstruction" if i == 0 else "")
        ax_s_recon.axis("off")

        if show_spectrum:
            inp_np = inputs[i]
            out_np = outputs[i]

            amp_list_i, _ = plot_fourier_transform_amplitude_phase(inp_np)
            amp_list_o, _ = plot_fourier_transform_amplitude_phase(out_np)

            for c in range(C):
                row_idx = 1 + c                
                spec_i = amp_list_i[c]
                spec_o = amp_list_o[c]

                spec_i, _ = equalize(spec_i)
                spec_o, _ = equalize(spec_o)

                ax_f_orig = axes[row_idx][col_orig]
                ax_f_orig.imshow(spec_i, cmap='inferno', origin='upper')
                
                if i == 0:
                    ax_f_orig.set_ylabel(f"Spectrum\nChannel {c}", fontsize=9)
                
                ax_f_orig.set_xticks([])
                ax_f_orig.set_yticks([])

                ax_f_recon = axes[row_idx][col_recon]
                ax_f_recon.imshow(spec_o, cmap='inferno', origin='upper')
                ax_f_recon.axis("off")

    return fig

def plot_classifications(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    dataset_type: str,
    class_names: Optional[List[str]] = None,
    num_samples: int = 5,
) -> plt.Figure:
    """
    Show input images with predicted and ground truth labels for classification tasks.

    Args:
        inputs: Input images tensor of shape (B, C, H, W)
        predictions: Predicted class indices of shape (B,)
        labels: Ground truth class indices of shape (B,)
        dataset_type: Type of dataset ("polsar", "sar", "grayscale", etc.)
        class_names: Optional list of class names mapping indices to strings.
        num_samples: Number of samples to display

    Returns:
        Matplotlib figure object containing the classified images.
    """
    B, C, H, W = inputs.shape
    num_samples = min(num_samples, B)

    # Create figure with 1 row and num_samples columns
    fig, axes = plt.subplots(1, num_samples, squeeze=False, constrained_layout=True, figsize=(num_samples * 3, 3.5))

    for s in range(num_samples):
        img_i = inputs[s, :]
        pred_idx = predictions[s].item()
        true_idx = labels[s].item()        
        img_i, cmap, origin, _ = _display_img(img_i, dataset_type)
        # --- Plotting ---
        ax = axes[0][s]
        ax.imshow(img_i, cmap=cmap, origin=origin)
        ax.axis("off")

        # --- Labeling ---
        pred_name = class_names[pred_idx] if class_names else str(pred_idx)
        true_name = class_names[true_idx] if class_names else str(true_idx)
        
        # Color code title: Green if correct, Red if wrong
        color = 'green' if pred_idx == true_idx else 'red'
        
        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=10, fontweight='bold')

    return fig

def plot_segmentations(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    dataset_type: str,
    number_classes: int,
    class_names: Optional[List[str]] = None,
    num_samples: int = 5,
) -> plt.Figure:
    """
    Visualize Input, Ground Truth, and Predicted Segmentation masks with a Legend.
    Displays samples in rows.

    Args:
        inputs: (B, C, H, W) Original images.
        predictions: (B, C, H, W) Logits OR (B, H, W) Indices.
        labels: (B, H, W) Ground truth indices.
        dataset_type: "polsar", "sar", or "grayscale".
        number_classes: Total number of segmentation classes.
        class_names: Optional list of class names. If None, uses "Class X".
        num_samples: Number of samples to display.
    """
    B, C, H, W = inputs.shape
    num_samples = min(num_samples, B)

    class_colors = {
        7: {
            0: "black",
            1: "purple",
            2: "blue",
            3: "green",
            4: "red",
            5: "cyan",
            6: "yellow",
        },
        5: {
            0: "black",
            1: "green",
            2: "brown",
            3: "blue",
            4: "yellow",
        },
    }.get(number_classes, {})
    # If predictions are logits, convert to class indices    
    # --- Colormap & Legend Setup ---
    # Determine number of classes based on data or provided names
    
    cmap = ListedColormap([class_colors[key] for key in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(class_colors))
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # --- Plotting ---
    # Create grid: Rows = Samples, Cols = 3 (Input, GT, Pred)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples), constrained_layout=True)
    
    if num_samples == 1:
        axes = axes[None, :]

    for s in range(num_samples):
        img_i = inputs[s]
        mask_true = labels[s].numpy()
        mask_pred = predictions[s].numpy()

        # Input Image Processing
        img_i, cmap_img, origin, _ = _display_img(img_i, dataset_type)

        # 1. Input Image
        ax_in = axes[s, 0]
        ax_in.imshow(img_i, cmap=cmap_img, origin=origin)
        ax_in.axis("off")
        
        # 2. Ground Truth Mask (using custom cmap/norm)
        ax_gt = axes[s, 1]
        ax_gt.imshow(mask_true, cmap=cmap, norm=norm, origin=origin, interpolation="nearest")
        ax_gt.axis("off")

        # 3. Prediction Mask (using custom cmap/norm)
        ax_pred = axes[s, 2]
        ax_pred.imshow(mask_pred, cmap=cmap, norm=norm, origin=origin, interpolation="nearest")
        ax_pred.axis("off")

        if s == 0:
            ax_in.set_title("Input", fontweight="bold")
            ax_gt.set_title("Ground Truth", fontweight="bold")
            ax_pred.set_title("Prediction", fontweight="bold")

        ax_in.text(-0.1, 0.5, f"Sample {s}", transform=ax_in.transAxes, 
                   va='center', ha='right', rotation=90, fontweight='bold')

    # Add the legend outside the plot area
    # bbox_to_anchor places it to the right of the last subplot
    fig.legend(
        handles=patches, 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5), 
        title="Classes", 
        borderaxespad=0.
    )
    return fig

def plot_pauli_decomposition(
    image1: np.ndarray,
    image2: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute and visualize the Pauli decomposition of a PolSAR image.

    Args:
        image1: Complex-valued original PolSAR image array
        image2: Complex-valued reconstructed PolSAR image array (or None to skip)
    Returns:
        Pauli decomposition image normalized to uint8 range [0, 255]
    """    
    image1 = pauli_transform(image1)
    if image2 is not None:
        image2 = pauli_transform(image2)
    
    if image2 is not None:
        num_cols = 2
    else:
        num_cols = 1
    fig, axes = plt.subplots(1, num_cols, squeeze=False, constrained_layout=True)
    image1, _ = equalize(image1)
    axes[0, 0].imshow(image1.transpose(1, 2, 0), origin="lower")
    axes[0, 0].axis("off")

    if image2 is not None:
        image2, _ = equalize(image2)
        axes[0, 1].imshow(image2.transpose(1, 2, 0), origin="lower")
        axes[0, 1].axis("off")

    return fig

def plot_krogager_decomposition(
    image1: np.ndarray,
    image2: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute and visualize the Krogager decomposition of a PolSAR image.

    Args:
        image1: Complex-valued original PolSAR image array
        image2: Complex-valued reconstructed PolSAR image array (or None to skip)
    Returns:
        Krogager decomposition image normalized to uint8 range [0, 255]
    """
    image1 = krogager_transform(image1)
    if image2 is not None:
        image2 = krogager_transform(image2)
    if image2 is not None:
        num_cols = 2
    else:
        num_cols = 1
    fig, axes = plt.subplots(1, num_cols, squeeze=False, constrained_layout=True)
    image1, _ = equalize(image1)
    axes[0, 0].imshow(image1.transpose(1, 2, 0), origin="lower")
    axes[0, 0].axis("off")

    if image2 is not None:
        image2, _ = equalize(image2)
        axes[0, 1].imshow(image2.transpose(1, 2, 0), origin="lower")
        axes[0, 1].axis("off")
    return fig

def plot_h_alpha_decomposition(
    image1: np.ndarray,
    image2: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute and visualize the H-alpha classification of a PolSAR image.
    Args:
        image1: Complex-valued original PolSAR image array
        image2: Complex-valued reconstructed PolSAR image array (or None to skip)
    Returns:
        H-alpha classification image normalized to uint8 range [0, 255]
    """
    # Define H-alpha class colors and names
    h_alpha_class_info = {
        1: {"color": "green", "name": "Complex structures"},
        2: {"color": "yellow", "name": "Random anisotropic scatterers"},
        4: {"color": "blue", "name": "Double reflection propagation effects"},
        5: {"color": "pink", "name": "Anisotropic particles"},
        6: {"color": "purple", "name": "Random surfaces"},
        7: {"color": "red", "name": "Dihedral reflector"},
        8: {"color": "brown", "name": "Dipole"},
        9: {"color": "gray", "name": "Bragg surface"},
    }
    h_alpha_class_colors = {k: v["color"] for k, v in h_alpha_class_info.items()}
    h_alpha_cmap = ListedColormap([i for i in h_alpha_class_colors.values()])
    h_alpha_bounds = list(h_alpha_class_colors.keys())
    h_alpha_norm = BoundaryNorm(h_alpha_bounds, len(h_alpha_class_colors))
    h_alpha_patches = [
        mpatches.Patch(
            color=h_alpha_class_info[i]["color"],
            label=f"{i}: {h_alpha_class_info[i]['name']}",
        )
        for i in h_alpha_class_info
    ]
    # Compute Pauli decomposition
    image1 = pauli_transform(image1)
    if image2 is not None:
        image2 = pauli_transform(image2)

    # Compute H-alpha classifications    
    image1 = h_alpha(image1)
    if image2 is not None:
        image2 = h_alpha(image2)
    if image2 is not None:
        num_cols = 2
    else:
        num_cols = 1

    fig, axes = plt.subplots(1, num_cols, constrained_layout=True)
    # Original H-alpha
    axes[0].imshow(
        image1, origin="lower", cmap=h_alpha_cmap, norm=h_alpha_norm
    )
    axes[0].axis("off")
    # Reconstructed H-alpha
    if image2 is not None:
        axes[1].imshow(
            image2, origin="lower", cmap=h_alpha_cmap, norm=h_alpha_norm
        )
        axes[1].axis("off")
        # Add legend
    fig.legend(
        handles=h_alpha_patches, bbox_to_anchor=(1.15, 0.8), loc="upper left"
    )
    return fig

def plot_h_alpha_plane(
    image1: np.ndarray,
    image2: np.ndarray,
) -> np.ndarray:
    """
    Compute and visualize the H-alpha plane of a PolSAR image.
    Args:
        image1: Complex-valued original PolSAR image array
        image2: Complex-valued reconstructed PolSAR image array
    Returns:
        Matplotlib figure object containing the H-alpha plane visualization.
    """
    h_alpha_class_info = {
        1: {"color": "green", "name": "Complex structures"},
        2: {"color": "yellow", "name": "Random anisotropic scatterers"},
        4: {"color": "blue", "name": "Double reflection propagation effects"},
        5: {"color": "pink", "name": "Anisotropic particles"},
        6: {"color": "purple", "name": "Random surfaces"},
        7: {"color": "red", "name": "Dihedral reflector"},
        8: {"color": "brown", "name": "Dipole"},
        9: {"color": "gray", "name": "Bragg surface"},
    }
    # Compute Pauli decomposition
    image1 = pauli_transform(image1)
    image2 = pauli_transform(image2)

    # Compute H-alpha classifications
    h_alpha_1 = h_alpha(image1)
    h_alpha_2 = h_alpha(image2)

    # Compute H-alpha coordinates
    son = 7
    h1, a1 = compute_h_alpha_coords(image1, son=son)
    h2, a2 = compute_h_alpha_coords(image2, son=son)

    assert h_alpha_1.shape == h_alpha_2.shape == h1.shape == h2.shape
    miscls_mask = (h_alpha_1 != h_alpha_2)
    idx_y, idx_x = np.where(miscls_mask)

    max_segments = 5000
    if idx_x.size > max_segments:
        sel = np.random.RandomState(0).choice(idx_x.size, size=max_segments, replace=False)
        idx_x, idx_y = idx_x[sel], idx_y[sel]

    # Now H on x-axis, α on y-axis
    segs = np.stack([
        np.stack([h1[idx_y, idx_x], a1[idx_y, idx_x]], axis=-1),
        np.stack([h2[idx_y, idx_x],  a2[idx_y, idx_x]],  axis=-1),
    ], axis=1)

    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 90)
    axes.grid(True, ls="--", alpha=0.4)

    # draw zones behind everything
    draw_halpha_zones(axes, h_alpha_class_info)

    alpha2, entropy2, alpha3, entropy3 = compute_physical_boundary()
    axes.plot(entropy2, np.degrees(alpha2),
                'k-', linewidth=2.0,
                label='Physical boundary', zorder=4)
    axes.plot(entropy3, np.degrees(alpha3),
                'k-', linewidth=2.0,
                zorder=4)

    axes.legend(frameon=True, loc="upper left")
    if segs.shape[0] == 0:
        axes.text(0.5, 0.5, "No misclassified pixels", ha="center", va="center", transform=axes.transAxes)
    else:
        lc = LineCollection(segs, linewidths=0.5, alpha=0.35)
        axes.add_collection(lc)

        axes.scatter(h1[idx_y, idx_x], a1[idx_y, idx_x],
                        s=6, marker='o', c='red',  label='Original', alpha=0.25)
        axes.scatter(h2[idx_y, idx_x],  a2[idx_y, idx_x],
                        s=6, marker='o', c='blue', label='Reconstructed', alpha=0.25)
        axes.legend(frameon=True, loc="upper left")
    return fig

def plot_cameron_decomposition(
    image1: np.ndarray,
    image2: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute and visualize the Cameron classification of a PolSAR image.
    Args:
        image1: Complex-valued original PolSAR image array
        image2: Complex-valued reconstructed PolSAR image array (or None to skip)
    Returns:
        Cameron classification image normalized to uint8 range [0, 255]
    """
    # Define Cameron class colors and names (classes 1-11)
    cameron_class_info = {
        1: {"color": "red", "name": "Non-reciprocal"},
        2: {"color": "orange", "name": "Asymmetric"},
        3: {"color": "yellow", "name": "Left helix"},
        4: {"color": "green", "name": "Right helix"},
        5: {"color": "blue", "name": "Symmetric"},
        6: {"color": "purple", "name": "Trihedral"},
        7: {"color": "brown", "name": "Dihedral"},
        8: {"color": "pink", "name": "Dipole"},
        9: {"color": "gray", "name": "Cylinder"},
        10: {"color": "olive", "name": "Narrow dihedral"},
        11: {"color": "cyan", "name": "Quarter-wave"},
    }
    cameron_class_colors = {k: v["color"] for k, v in cameron_class_info.items()}
    cameron_cmap = ListedColormap([i for i in cameron_class_colors.values()])
    cameron_bounds = list(cameron_class_colors.keys())
    cameron_norm = BoundaryNorm(cameron_bounds, len(cameron_class_colors))
    cameron_patches = [
        mpatches.Patch(
            color=cameron_class_info[i]["color"],
            label=f"{i}: {cameron_class_info[i]['name']}",
        )
        for i in cameron_class_info
    ]

    # Compute Cameron classifications
    image1 = cameron(image1)
    if image2 is not None:
        image2 = cameron(image2)
    if image2 is not None:
        num_cols = 2
    else:
        num_cols = 1
    fig, axes = plt.subplots(1, num_cols, constrained_layout=True)

    # Original Cameron
    axes[0].imshow(
        image1, origin="lower", cmap=cameron_cmap, norm=cameron_norm
    )
    axes[0].axis("off")

    # Reconstructed Cameron
    if image2 is not None:
        axes[1].imshow(
            image2, origin="lower", cmap=cameron_cmap, norm=cameron_norm
        )
        axes[1].axis("off")
        # Add legend
    fig.legend(
        handles=cameron_patches, bbox_to_anchor=(1.15, 0.8), loc="upper left"
    )
    return fig

def create_dataset_split_mask(
    cfg: Dict[str, Any],
    full_loader: torch.utils.data.DataLoader,
    train_indices: List[int],
    valid_indices: List[int],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    test_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create dataset split mask (0=Train, 1=Valid, 2=Test).
    Args:
        cfg: Configuration dictionary
        full_loader: DataLoader for the full dataset
        train_indices: List of training indices
        valid_indices: List of validation indices
        nsamples_per_rows: Number of samples per row
        nsamples_per_cols: Number of samples per column
        test_indices: Optional list of test indices
    Returns:
        2D numpy array mask with values 0 (Train), 1 (Valid), 2 (Test), -1 (No data)
    """
    patch_size = cfg["data"]["dataset"]["patch_size"]
    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    # Create mask initialized to -1 (no data)
    mask = np.full((nb_rows, nb_cols), -1, dtype=np.int32)

    # Collect all indices from loader
    correct_indice_tensors = []
    for data in full_loader:
        # Extract indices (usually the last element in the batch tuple)
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            correct_indice_tensors.extend(data[-1].cpu().detach().numpy())

    # Create sets for quick lookup
    sets_indices = [set(train_indices), set(valid_indices)]
    if test_indices is not None:
        sets_indices.append(set(test_indices))

    # Place each patch
    for real_index in correct_indice_tensors:
        row = real_index // nsamples_per_cols
        col = real_index % nsamples_per_cols

        if row >= nsamples_per_rows or col >= nsamples_per_cols:
            continue

        h_start = row * patch_size
        w_start = col * patch_size

        if real_index in sets_indices[0]:
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 0
        elif real_index in sets_indices[1]:
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 1
        elif len(sets_indices) > 2 and real_index in sets_indices[2]:
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 2

    return mask


def plot_dataset_split_mask(
    mask: np.ndarray,
    patch_size: int,
    train_indices: List[int],
    valid_indices: List[int],
    test_indices: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Create a spatial visualization of how the dataset is split into train/validation/test sets.
    Uses a patch-based legend style instead of a colorbar.
    Args:
        mask: 2D numpy array mask with values 0 (Train), 1 (Valid), 2 (Test), -1 (No data)
        patch_size: Size of each patch
        train_indices: List of training indices
        valid_indices: List of validation indices
        test_indices: Optional list of test indices
    Returns:
        Matplotlib Figure object with the visualization
    """
    # Creation of the figure (Image on the left, Stats/Legend on the right)
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, constrained_layout=True)
    ax_img = axes[0]
    ax_stats = axes[1]

    # 1. DDefinition of colors and labels
    # Configuration dictionary similar to "Cameron" logic
    split_info = {
        0: {"color": "#1f77b4", "name": "Train"},       # Blue
        1: {"color": "#ff7f0e", "name": "Validation"},  # Orange
        2: {"color": "#2ca02c", "name": "Test"},        # Green
    }
    
    # DDetermine how many sets are used
    active_indices = [0, 1]
    if test_indices is not None:
        active_indices.append(2)

    # 2. Creation of "Patches" for the legend (Cameron logic)
    legend_patches = [
        mpatches.Patch(
            color=split_info[i]["color"],
            label=split_info[i]["name"]
        )
        for i in active_indices
    ]

    # Preparation of the colormap for the image
    colors = [split_info[i]["color"] for i in active_indices]
    cmap = ListedColormap(colors)
    bounds = list(range(len(active_indices) + 1))
    norm = BoundaryNorm(bounds, len(bounds) - 1)

    # 3. Displaying the image (WITHOUT colorbar)
    ax_img.imshow(mask, cmap=cmap, norm=norm, origin="lower", aspect="equal")

    # Grid
    for h in range(0, mask.shape[0], patch_size):
        ax_img.axhline(y=h, color="white", linewidth=0.5, alpha=0.3)
    for w in range(0, mask.shape[1], patch_size):
        ax_img.axvline(x=w, color="white", linewidth=0.5, alpha=0.3)

    # 4. Configuration of the statistics area (right)
    ax_stats.axis('off')

    # A. Adding the Legend (at the top of the right column)
    # loc='upper left' places the legend at the upper left of the stats area
    ax_stats.legend(
        handles=legend_patches, 
        loc='upper left', 
        bbox_to_anchor=(0.0, 1.0), # Anchored at the very top
        fontsize=10,
        title="Dataset Split"
    )

    # B. Adding statistics text below the legend
    stats_text = f"Train patches: {len(train_indices)}\n"
    stats_text += f"Validation patches: {len(valid_indices)}\n"
    if test_indices is not None:
        stats_text += f"Test patches: {len(test_indices)}\n"
    stats_text += f"Patch size: {patch_size}×{patch_size}\n"
    stats_text += f"Patch stride: {patch_size}\n"
    stats_text += f"Image size: {mask.shape[0]}×{mask.shape[1]}"

    ax_stats.text(
        0.0,    # x
        0.5,    # y (Vertical centering)
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return fig

def plot_segmentation_full_image(
    ground_truth_segmentation: np.ndarray,
    predicted_segmentation: np.ndarray,
    original_image: np.ndarray,
    dataset_type: str,
    number_classes: int,
    ignore_index: Optional[int] = None,
) -> None:
    """
    Create side-by-side visualization of full image segmentation: Original Image | Ground Truth | Prediction.

    Args:
        ground_truth_segmentation: Full ground truth segmentation image
        predicted_segmentation: Full predicted segmentation image
        original_image: Original SAR image (complex-valued, shape: [3, height, width])
        dataset_type: Type of dataset (e.g., "sar", "optical")
        number_classes: Number of segmentation classes
        ignore_index: Index to ignore in segmentation (optional)
    """
    # Set up segmentation colormap
    class_colors = {
        7: {
            0: "black",
            1: "purple",
            2: "blue",
            3: "green",
            4: "red",
            5: "cyan",
            6: "yellow",
        },
        5: {
            0: "black",
            1: "green",
            2: "brown",
            3: "blue",
            4: "yellow",
        },
    }.get(number_classes, {})

    cmap = ListedColormap([class_colors[key] for key in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(class_colors))
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # Mask prediction if ignore_index is provided
    if ignore_index is not None:
        masked_predicted = predicted_segmentation.copy()
        masked_predicted[ground_truth_segmentation == ignore_index] = ignore_index
    else:
        masked_predicted = predicted_segmentation

    # Create four-panel visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 8), constrained_layout=True)

    if dataset_type == "polsar":
        original_image = pauli_transform(original_image)
        original_image, _ = equalize(original_image)
        original_image = original_image.transpose(1, 2, 0)

    # Original Image (adaptive based on dataset type)
    axes[0].imshow(original_image, origin="lower")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(ground_truth_segmentation, cmap=cmap, norm=norm, origin="lower")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(predicted_segmentation, cmap=cmap, norm=norm, origin="lower")
    axes[2].axis("off")

    axes[3].imshow(masked_predicted, cmap=cmap, norm=norm, origin="lower")
    axes[3].axis("off")

    # Add legend to the figure
    plt.figlegend(
        handles=patches, loc="center right", title="Classes", bbox_to_anchor=(1.02, 0.5)
    )

    plt.suptitle("Full Image Segmentation Analysis", fontsize=16, fontweight="bold")
    
    return fig
    

def plot_fourier_transform_amplitude_phase(image):
    amplitude_ft_images = []
    phase_ft_vectors = []

    for channel in range(image.shape[0]):
        fft_img = np.fft.fftshift(np.fft.fft2(image[channel, :, :]))
        amplitude = np.abs(fft_img)
        phase = np.angle(fft_img)

        amplitude_ft_images.append(amplitude)
        phase_ft_vectors.append(phase)

    return amplitude_ft_images, phase_ft_vectors

def _plot_confusion_matrix_subplot(ax, cm: np.ndarray, labels: List[Any]):
    """Helper to plot a nice confusion matrix on a given axis."""
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="hot",        # Changed to 'hot'
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 8},
        square=True        # Forces the heatmap cells to be square
    )
    
    ax.tick_params(axis='both', which='major', labelsize=8)

def _plot_per_class_metrics_subplot(ax, per_class_metrics: Dict[int, Dict[str, float]], labels: List[Any]):
    """Helper to plot grouped bar chart for Precision/Recall/F1 with rotated labels."""
    metrics_names = ["precision", "recall", "f1_score"]
    x = np.arange(len(labels))
    width = 0.25
    multiplier = 0

    # Define colors manually to ensure distinctness
    color_map = {"precision": "skyblue", "recall": "lightgreen", "f1_score": "salmon"}

    for metric in metrics_names:
        offset = width * multiplier
        # Use explicit list comprehension over the keys to match the 'labels' order
        # Assuming per_class_metrics keys match the order of 'labels' passed in
        values = [per_class_metrics[k].get(metric, 0.0) for k in per_class_metrics]
        ax.bar(x + offset, values, width, label=metric.capitalize(), color=color_map.get(metric))
        multiplier += 1

    # 1. Set the tick positions (centered on the group of bars)
    ax.set_xticks(x + width)
    
    # 2. Set the labels with rotation and alignment
    # rotation_mode="anchor" + ha="right" keeps the text end aligned with the tick
    is_text_label = any(isinstance(l, str) and len(l) > 2 for l in labels)
    
    if is_text_label:
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, 1.05)
    
    # Legend at top right
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='y', which='major', labelsize=8)

def plot_reconstruction_error_analysis(
    errors: Dict[str, np.ndarray],
) -> plt.Figure:
    """
    Plot histograms of reconstruction errors.
    
    Args:
        errors: Dict with keys like 'amplitude_error', 'phase_error', 'angular_distance'.
                Values are flattened numpy arrays of errors.
    """
    valid_keys = [k for k in errors.keys() if errors[k] is not None and errors[k].size > 0]
    n_plots = len(valid_keys)
    
    if n_plots == 0:
        fig = plt.figure()
        plt.text(0.5, 0.5, "No error data provided")
        return fig

    fig, axes = plt.subplots(1, n_plots, constrained_layout=True)
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, valid_keys):
        data = errors[key]
        # Robust limits to ignore outliers
        low, high = np.percentile(data, [1, 99])
        data_filtered = data[(data >= low) & (data <= high)]
        
        sns.histplot(data_filtered, kde=True, ax=ax, color='purple', alpha=0.6)
        ax.grid(True, alpha=0.2)
    return fig

def plot_classification_metrics(
    metrics: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None,
    ignore_index: Optional[int] = None
) -> plt.Figure:
    """
    Generic dashboard for classification metrics.
    Works for: Standard Classification, H-Alpha, Cameron.
    
    Args:
        metrics: Dictionary returned by metrics_registry.
        class_names: Optional mapping {class_id: "Class Name"}.
        ignore_index: Optional class ID to exclude from visualization (e.g. background).
        
    Returns:
        Matplotlib Figure.
    """
    # Use constrained_layout to handle spacing automatically
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    
    # Adjust width ratios to give the bar chart more space, but keep CM square
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # --- 1. Data Extraction & Filtering ---
    cm = np.array(metrics.get("confusion_matrix_normalized", []))
    raw_labels = metrics.get("class_labels", [])
    per_class = metrics.get("per_class_metrics", {}).copy() # Copy to avoid mutating original

    # Convert labels to list if it's a dict or other iterable
    if isinstance(raw_labels, dict):
        labels = list(raw_labels.keys())
    else:
        labels = list(raw_labels)

    # Filter out ignore_index if provided and present
    if ignore_index is not None and ignore_index in labels:
        # Find the position (index) of the class ID in the labels list
        idx_to_remove = labels.index(ignore_index)
        
        # 1. Remove from labels list
        labels.pop(idx_to_remove)
        
        # 2. Remove from Confusion Matrix (rows and cols)
        if cm.size > 0:
            cm = np.delete(cm, idx_to_remove, axis=0) # Remove row
            cm = np.delete(cm, idx_to_remove, axis=1) # Remove col
            
        # 3. Remove from per-class metrics
        if ignore_index in per_class:
            del per_class[ignore_index]

    # --- 2. Label Preparation ---
    # Create display labels only for the remaining classes
    display_labels = [class_names.get(k, str(k)) if class_names else str(k) for k in labels]

    # --- 3. Confusion Matrix (Left) ---
    ax_cm = fig.add_subplot(gs[0, 0])
    
    if cm.size > 0:
        _plot_confusion_matrix_subplot(ax_cm, cm, display_labels)
    else:
        ax_cm.text(0.5, 0.5, "No Confusion Matrix", ha='center')
        ax_cm.axis("off")

    # --- 4. Per-Class Metrics (Right) ---
    ax_bar = fig.add_subplot(gs[0, 1])
    
    if per_class:
        # Filter per_class to match the order of *remaining* labels
        ordered_per_class = {l: per_class[l] for l in labels if l in per_class}
        _plot_per_class_metrics_subplot(ax_bar, ordered_per_class, display_labels)
    else:
        ax_bar.text(0.5, 0.5, "No Per-Class Metrics", ha='center')
        ax_bar.axis("off")

    return fig