# Standard library imports
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party imports
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d


def pauli_transform(img: np.ndarray) -> np.ndarray:
    """
    Apply Pauli decomposition to PolSAR data.

    Args:
        img: Complex SAR image with shape (3, height, width) representing [HH, HV, VV]

    Returns:
        Pauli decomposed image with shape (3, height, width) representing [k1, k2, k3]

    Raises:
        ValueError: If input shape is invalid
    """
    if img.shape[0] == 4:
        img = np.stack(
            (
                img[0, :, :],  # HH
                (img[1, :, :] + img[2, :, :]) / 2,  # HV + VH
                img[3, :, :],  # VV (assuming 4th channel is VV)
            ),
            axis=0,
        )
    elif img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (HH, HV, VV), got {img.shape[0]}")

    hh = img[0, :, :]
    hv = img[1, :, :]
    vv = img[2, :, :]

    # Pauli basis vectors: k1 = (S_HH - S_VV)/√2, k2 = 2*S_HV/√2, k3 = (S_HH + S_VV)/√2
    sqrt_2_inv = 1.0 / np.sqrt(2.0)
    return sqrt_2_inv * np.stack(
        (
            hh - vv,
            2 * hv,
            hh + vv,
        ),
        dtype=np.complex64,
    )


def krogager_transform(img: np.ndarray) -> np.ndarray:
    """
    Apply Krogager decomposition to PolSAR data.

    Args:
        img: Complex SAR image with shape (3, height, width) representing [HH, HV, VV]

    Returns:
        Krogager decomposed image with shape (3, height, width) representing [kd, kh, ks]

    Raises:
        ValueError: If input shape is invalid
    """
    if img.shape[0] == 4:
        img = np.stack(
            (
                img[0, :, :],  # HH
                (img[1, :, :] + img[2, :, :]) / 2,  # HV + VH
                img[3, :, :],  # VV (assuming 4th channel is VV)
            ),
            axis=0,
        )
    if img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (HH, HV, VV), got {img.shape[0]}")

    hh = img[0, :, :]
    hv = img[1, :, :]
    vv = img[2, :, :]

    # Krogager decomposition: sphere, diplane, helix components
    rr = 1j * hv + 0.5 * (hh - vv)
    ll = 1j * hv - 0.5 * (hh - vv)
    rl = 0.5j * (hh + vv)  # Fixed coefficient

    return np.stack(
        (
            np.minimum(np.abs(rr), np.abs(ll)),  # kd: sphere component
            np.abs(np.abs(rr) - np.abs(ll)),  # kh: diplane component
            np.abs(rl),  # ks: helix component
        ),
        dtype=np.float32,  # Real-valued output
    )


def compute_h_alpha_coords(fullsamples: np.ndarray, son: int = 7, eps: float = 1e-10):
    """
    Vectorized H (entropy in [0,1]) and α (deg in [0,90]) on a sliding son×son window,
    matching the covariance/eigendecomp in _compute_classes_h_alpha.
    Returns: H_mat (H_out×W_out), alpha_mat (H_out×W_out)
    """
    p, H, W = fullsamples.shape
    H_out, W_out = H - (son - 1), W - (son - 1)

    kernel = np.ones((son, son), dtype=fullsamples.dtype)
    cov = np.empty((H_out, W_out, p, p), dtype=complex)
    for i in range(p):
        for j in range(p):
            prod = fullsamples[i, ...] * np.conj(fullsamples[j, ...])
            cov[..., i, j] = convolve2d(prod, kernel, mode="valid") / (son**2)

    eigvals, eigvecs = np.linalg.eigh(cov)
    p_vec = eigvals / (eigvals.sum(axis=-1, keepdims=True) + eps)

    # H in [0,1]
    H_mat = -np.sum(p_vec * (np.log(p_vec + eps) / np.log(3)), axis=-1)
    H_mat = np.clip(H_mat, 0.0, 1.0)

    # α (Cloude–Pottier): weight eigenvector angles by p_vec
    alpha_vec = np.arccos(np.clip(np.abs(eigvecs[..., 0, :]), 0.0, 1.0))
    alpha_mat = np.sum(p_vec * alpha_vec, axis=-1) * (180.0 / np.pi)
    alpha_mat = np.clip(alpha_mat, 0.0, 90.0)
    return H_mat, alpha_mat

def draw_halpha_zones(ax, class_info: Dict[int, Dict[str, str]]):
    """
    Draw the 9 standard Cloude–Pottier H–α zones as translucent rectangles.
    Axes: x = Entropy H ∈ [0,1], y = α ∈ [0,90] (degrees).
    Colors taken from class_info[id]["color"].
    """
    # Entropy band edges
    H_L, H_M, H_H, H_MAX = 0.0, 0.5, 0.9, 1.0

    # Alpha thresholds per band (deg)
    # Low entropy: 0–0.5
    th_L = [0.0, 42.5, 47.5, 90.0]     # → classes [9, 8, 7]
    # Medium entropy: 0.5–0.9
    th_M = [0.0, 40.0, 50.0, 90.0]     # → classes [6, 5, 4]
    # High entropy: 0.9–1.0
    th_H = [0.0, 40.0, 55.0, 90.0]           # → classes [3, 2, 1]

    zones = [
        # (Hmin, Hmax, αmin, αmax, class_id)
        # Low entropy band
        (H_L, H_M, th_L[0], th_L[1], 9),
        (H_L, H_M, th_L[1], th_L[2], 8),
        (H_L, H_M, th_L[2], th_L[3], 7),
        # Medium entropy band
        (H_M, H_M + (H_H - H_M), th_M[0], th_M[1], 6),
        (H_M, H_H,                 th_M[1], th_M[2], 5),
        (H_M, H_H,                 th_M[2], th_M[3], 4),
        # High entropy band
        (H_H, H_MAX, th_H[0], th_H[1], 3),
        (H_H, H_MAX, th_H[1], th_H[2], 2),
        (H_H, H_MAX, th_H[1], th_H[2], 1),
    ]

    # Draw rectangles
    for Hmin, Hmax, amin, amax, cid in zones:
        color = class_info.get(cid, {}).get("color", "lightgray")
        rect = Rectangle(
            (Hmin, amin), Hmax - Hmin, amax - amin,
            facecolor=color, edgecolor=color, alpha=0.12, linewidth=0.8, zorder=0
        )
        ax.add_patch(rect)
        # annotate class id
        ax.text(
            (Hmin + Hmax) / 2.0, (amin + amax) / 2.0, str(cid),
            ha="center", va="center", fontsize=8, alpha=0.6, zorder=1
        )

    # Gridlines at canonical boundaries
    for x in [0.5, 0.9]:
        ax.axvline(x, ls="--", lw=0.8, alpha=0.4, zorder=2)
    for y in [40.0, 42.5, 47.5, 50.0, 55.0]:
        ax.axhline(y, ls="--", lw=0.8, alpha=0.4, zorder=2)
    
def compute_physical_boundary():
    """
    Compute the physically feasible boundary of the Cloude–Pottier H–α plane.
    Returns (entropy_all, alpha_all_deg).
    """
    m2 = np.arange(0, 1.01, 0.01)
    entropy2 = []
    alpha2 = []
    entropy3 = []
    alpha3 = []

    for i in range(101):
        T = np.array([[1, 0, 0],
                    [0, m2[i], 0],
                    [0, 0, m2[i]]])
    
        D, V = np.linalg.eig(T)
        tr = np.sum(D)
        P = D / tr + np.finfo(float).eps  # Avoid log(0)
    
        alpha2_val = np.sum(P * np.arccos(np.abs(V[0, :])))
        entropy2_val = -np.sum(P * np.log10(P)) / np.log10(3)
    
        alpha2.append(alpha2_val)
        entropy2.append(entropy2_val)
    
        if i < 50:
            T2 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 2*m2[i]]])
        else:
            T2 = np.array([[2*m2[i]-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
        D2, V2 = np.linalg.eig(T2)
        tr2 = np.sum(D2)
        P2 = D2 / tr2 + np.finfo(float).eps
    
        alpha3_val = np.sum(P2 * np.arccos(np.abs(V2[0, :])))
        entropy3_val = -np.sum(P2 * np.log10(P2)) / np.log10(3)
    
        alpha3.append(alpha3_val)
        entropy3.append(entropy3_val)
    return alpha2, entropy2, alpha3, entropy3


def h_alpha(fullsamples: np.ndarray, son: int = 7, eps: float = 1e-10) -> np.ndarray:
    """
    Vectorized H–α classification using convolution and batched eigendecomposition.
    """
    if fullsamples.ndim != 3 or fullsamples.shape[0] != 3:
        raise ValueError(f"Expected shape (3, H, W), got {fullsamples.shape}")
    
    p, H, W = fullsamples.shape
    H_out, W_out = H - (son - 1), W - (son - 1)

    # 1) Build local covariance via convolution
    kernel = np.ones((son, son), dtype=fullsamples.dtype)
    cov = np.empty((H_out, W_out, p, p), dtype=complex)
    for i in range(p):
        for j in range(p):
            prod = fullsamples[i, ...] * np.conj(fullsamples[j, ...])
            cov[..., i, j] = convolve2d(prod, kernel, mode="valid") / (son**2)

    # 2) Batched eigendecomposition (last two dims)
    eigvals, eigvecs = np.linalg.eigh(cov)
    p_vec = eigvals / eigvals.sum(axis=-1, keepdims=True) + eps  # Avoid division by zero

    # 3) Entropy H
    H_mat = -np.sum(p_vec * np.log(p_vec), axis=-1)
    H_mat = np.clip(H_mat, 0, 1)

    # 4) Mean scattering angle α
    alpha_vec = np.arccos(np.clip(np.abs(eigvecs[..., 0, :]), 0, 1))
    alpha_mat = np.sum(p_vec * alpha_vec, axis=-1) * (180.0 / np.pi)
    alpha_mat = np.clip(alpha_mat, 0, 90)

    # 5) Vectorized classification rules
    classes = np.zeros((H_out, W_out), dtype=int)

    # H <= 0.5
    m = H_mat <= 0.5
    classes[m & (alpha_mat <= 42.5)] = 9
    classes[m & (alpha_mat > 42.5) & (alpha_mat <= 47.5)] = 8
    classes[m & (alpha_mat > 47.5)] = 7

    # 0.5 < H <= 0.9
    m = (H_mat > 0.5) & (H_mat <= 0.9)
    classes[m & (alpha_mat <= 40)] = 6
    classes[m & (alpha_mat > 40) & (alpha_mat <= 50)] = 5
    classes[m & (alpha_mat > 50)] = 4

    # 0.9 < H <= 1.0
    m = (H_mat > 0.9) & (H_mat <= 1.0)
    classes[m & (alpha_mat <= 55)] = 2
    classes[m & (alpha_mat > 55)] = 1

    return classes
    
def cameron_transform(SAR_img: np.ndarray, eps: float = 1e-10) -> List[np.ndarray]:
    """
    Compute Cameron decomposition parameters from PolSAR image.

    This function implements the Cameron coherent polarimetric decomposition
    based on the method described in Cameron, Youssef, and Leung (1996).
    The decomposition extracts symmetric and asymmetric scattering components.

    Args:
        SAR_img: Complex SAR image with shape (3, H, W) for [HH, HV, VV]
                 or (4, H, W) for [HH, HV, VH, VV]

    Returns:
        List of Cameron decomposition parameters:
        - For 3-pol data: 9 parameters [S_max1, S_max2, S_max4, S_min1, S_min2, S_min4, a, Tau, Psi_D]
        - For 4-pol data: 13 parameters [S_max1, S_max2, S_max3, S_max4, S_min1, S_min2, S_min3, S_min4, S_nr, a, Tau, Theta_rec, Psi_D]
        - For test compatibility: Always returns 13 parameters (padding with zeros for 3-pol)

    Raises:
        ValueError: If input doesn't have expected 3 or 4 channel format
    """
    if SAR_img.shape[0] not in [3, 4]:
        raise ValueError(f"Expected 3 channels, got {SAR_img.shape[0]}")

    is_full_pol = SAR_img.shape[0] == 4

    if is_full_pol:
        # 4-pol case: [HH, HV, VH, VV]
        S_hh = SAR_img[0]
        S_hv = SAR_img[1]
        S_vh = SAR_img[2]
        S_vv = SAR_img[3]

        # Compute norm of scattering vector (full-pol)
        a = np.sqrt(
            np.abs(S_hh) ** 2
            + np.abs(S_hv) ** 2
            + np.abs(S_vh) ** 2
            + np.abs(S_vv) ** 2
            + eps # Avoid division by zero
        )

        # Determine Pauli parameters (full-pol)
        alpha = (S_hh + S_vv) / np.sqrt(2)
        beta = (S_hh - S_vv) / np.sqrt(2)
        gamma = (S_hv + S_vh) / np.sqrt(2)
        delta = (S_vh - S_hv) / np.sqrt(2)

        # Determine parameter x
        numerator = beta * np.conj(gamma) + np.conj(beta) * gamma
        denominator_sq = numerator**2 + (np.abs(beta) ** 2 - np.abs(gamma) ** 2) ** 2

        # Avoid division by zero
        sin_x = np.zeros_like(numerator, dtype=complex)
        cos_x = np.zeros_like(numerator, dtype=complex)

        sin_x = numerator / np.sqrt(denominator_sq)
        cos_x = (
            np.abs(beta) ** 2 - np.abs(gamma) ** 2
        ) / np.sqrt(denominator_sq)

        # Compute angle x
        x = np.zeros_like(sin_x, dtype=float)
        sin_x_real = np.real(sin_x)
        cos_x_real = np.real(cos_x)

        # Handle different quadrants
        mask1 = sin_x_real >= 0
        mask2 = (sin_x_real < 0) & (cos_x_real >= 0)
        mask3 = (sin_x_real < 0) & (cos_x_real < 0)

        x[mask1] = np.arccos(np.clip(cos_x_real[mask1], -1, 1))
        x[mask2] = np.arcsin(np.clip(sin_x_real[mask2], -1, 1))
        x[mask3] = -np.arcsin(np.clip(sin_x_real[mask3], -1, 1)) - np.pi

        # Determine DS (dominant symmetric component) - full-pol
        scalar = (
            S_hh * np.cos(x / 2)
            + S_hv * np.sin(x / 2)
            + S_vh * np.sin(x / 2)
            - S_vv * np.cos(x / 2)
        ) / np.sqrt(2)

        DS_1 = (alpha + np.cos(x / 2) * scalar) / np.sqrt(2)
        DS_2 = np.sin(x / 2) * scalar / np.sqrt(2)
        DS_3 = np.sin(x / 2) * scalar / np.sqrt(2)
        DS_4 = (alpha - np.cos(x / 2) * scalar) / np.sqrt(2)

        # Normalize S_max
        S_max_norm = np.sqrt(
            np.abs(DS_1) ** 2
            + np.abs(DS_2) ** 2
            + np.abs(DS_3) ** 2
            + np.abs(DS_4) ** 2
            + eps # Avoid division by zero
        )

        S_max1 = DS_1 / S_max_norm
        S_max2 = DS_2 / S_max_norm
        S_max3 = DS_3 / S_max_norm
        S_max4 = DS_4 / S_max_norm

        # Determine S_rec (reciprocal component) - full-pol
        S_rec1 = S_hh
        S_rec2 = (S_hv + S_vh) / 2
        S_rec3 = (S_hv + S_vh) / 2
        S_rec4 = S_vv

        # Determine DS_rec
        scalar_rec = (
            S_rec1 * np.cos(x / 2)
            + S_rec2 * np.sin(x / 2)
            + S_rec3 * np.sin(x / 2)
            - S_rec4 * np.cos(x / 2)
        ) / np.sqrt(2)

        DS_rec1 = (alpha + np.cos(x / 2) * scalar_rec) / np.sqrt(2)
        DS_rec2 = np.sin(x / 2) * scalar_rec / np.sqrt(2)
        DS_rec3 = np.sin(x / 2) * scalar_rec / np.sqrt(2)
        DS_rec4 = (alpha - np.cos(x / 2) * scalar_rec) / np.sqrt(2)

        # Determine S_min
        S_min1_unnorm = S_rec1 - DS_rec1
        S_min2_unnorm = S_rec2 - DS_rec2
        S_min3_unnorm = S_rec3 - DS_rec3
        S_min4_unnorm = S_rec4 - DS_rec4

        S_min_norm = np.sqrt(
            np.abs(S_min1_unnorm) ** 2
            + np.abs(S_min2_unnorm) ** 2
            + np.abs(S_min3_unnorm) ** 2
            + np.abs(S_min4_unnorm) ** 2
            + eps # Avoid division by zero
        )

        S_min1 = S_min1_unnorm / S_min_norm
        S_min2 = S_min2_unnorm / S_min_norm
        S_min3 = S_min3_unnorm / S_min_norm
        S_min4 = S_min4_unnorm / S_min_norm

        # S_nr (non-reciprocal component) - full-pol only
        S_nr = np.divide(
            delta, np.abs(delta) + eps, out=np.zeros_like(delta),
        )

        # Theta_rec (reciprocity angle) - full-pol only
        S_rec_norm_for_theta = np.sqrt(
            np.abs(S_rec1) ** 2
            + np.abs(S_rec2) ** 2
            + np.abs(S_rec3) ** 2
            + np.abs(S_rec4) ** 2
            + eps # Avoid division by zero
        )
        Theta_rec = np.arccos(
            np.clip(S_rec_norm_for_theta / a, 0, 1)
        )

    else:
        # 3-pol case: [HH, HV, VV]
        S_hh = SAR_img[0]
        S_hv = SAR_img[1]
        S_vv = SAR_img[2]

        # Compute norm of scattering vector (3-pol)
        a = np.sqrt(np.abs(S_hh) ** 2 + 2 * np.abs(S_hv) ** 2 + np.abs(S_vv) ** 2)

        # Determine Pauli parameters (3-pol)
        alpha = (S_hh + S_vv) / np.sqrt(2)
        beta = (S_hh - S_vv) / np.sqrt(2)
        gamma = 2 * S_hv / np.sqrt(2)

        # Determine parameter x
        numerator = beta * np.conj(gamma) + np.conj(beta) * gamma
        denominator_sq = numerator**2 + (np.abs(beta) ** 2 - np.abs(gamma) ** 2) ** 2

        # Avoid division by zero
        sin_x = np.zeros_like(numerator, dtype=complex)
        cos_x = np.zeros_like(numerator, dtype=complex)

        sin_x = numerator / np.sqrt(denominator_sq + eps)  # Avoid division by zero
        cos_x = (
            np.abs(beta) ** 2 - np.abs(gamma) ** 2
        ) / np.sqrt(denominator_sq + eps)  # Avoid division by zero

        # Compute angle x
        x = np.zeros_like(sin_x, dtype=float)
        sin_x_real = np.real(sin_x)
        cos_x_real = np.real(cos_x)

        # Handle different quadrants
        mask1 = sin_x_real >= 0
        mask2 = (sin_x_real < 0) & (cos_x_real >= 0)
        mask3 = (sin_x_real < 0) & (cos_x_real < 0)

        x[mask1] = np.arccos(np.clip(cos_x_real[mask1], -1, 1))
        x[mask2] = np.arcsin(np.clip(sin_x_real[mask2], -1, 1))
        x[mask3] = -np.arcsin(np.clip(sin_x_real[mask3], -1, 1)) - np.pi

        # Determine DS (dominant symmetric component) - 3-pol
        scalar = (
            S_hh * np.cos(x / 2) + 2 * S_hv * np.sin(x / 2) - S_vv * np.cos(x / 2)
        ) / np.sqrt(2)

        DS_1 = (alpha + np.cos(x / 2) * scalar) / np.sqrt(2)
        DS_2 = np.sin(x / 2) * scalar / np.sqrt(2)
        DS_3 = np.sin(x / 2) * scalar / np.sqrt(2)
        DS_4 = (alpha - np.cos(x / 2) * scalar) / np.sqrt(2)

        # Normalize S_max
        S_max_norm = np.sqrt(
            np.abs(DS_1) ** 2
            + np.abs(DS_2) ** 2
            + np.abs(DS_3) ** 2
            + np.abs(DS_4) ** 2
            + eps  # Avoid division by zero
        )

        S_max1 = DS_1 / S_max_norm
        S_max2 = DS_2 / S_max_norm
        S_max3 = DS_3 / S_max_norm  # Same as S_max2 for 3-pol
        S_max4 = DS_4 / S_max_norm

        # Determine S_rec (reciprocal component) - 3-pol
        S_rec1 = S_hh
        S_rec2 = S_hv
        S_rec3 = S_hv  # Same as S_rec2 for 3-pol
        S_rec4 = S_vv

        # Determine DS_rec
        scalar_rec = (
            S_rec1 * np.cos(x / 2)
            + S_rec2 * np.sin(x / 2)
            + S_rec3 * np.sin(x / 2)
            - S_rec4 * np.cos(x / 2)
        ) / np.sqrt(2)

        DS_rec1 = (alpha + np.cos(x / 2) * scalar_rec) / np.sqrt(2)
        DS_rec2 = np.sin(x / 2) * scalar_rec / np.sqrt(2)
        DS_rec3 = np.sin(x / 2) * scalar_rec / np.sqrt(2)
        DS_rec4 = (alpha - np.cos(x / 2) * scalar_rec) / np.sqrt(2)

        # Determine S_min
        S_min1_unnorm = S_rec1 - DS_rec1
        S_min2_unnorm = S_rec2 - DS_rec2
        S_min3_unnorm = S_rec3 - DS_rec3
        S_min4_unnorm = S_rec4 - DS_rec4

        S_min_norm = np.sqrt(
            np.abs(S_min1_unnorm) ** 2
            + np.abs(S_min2_unnorm) ** 2
            + np.abs(S_min3_unnorm) ** 2
            + np.abs(S_min4_unnorm) ** 2
            + eps # Avoid division by zero
        )

        S_min1 = S_min1_unnorm / S_min_norm
        S_min2 = S_min2_unnorm / S_min_norm
        S_min3 = S_min3_unnorm / S_min_norm
        S_min4 = S_min4_unnorm / S_min_norm

        # For 3-pol data, S_nr and Theta_rec are zero (reciprocal case)
        S_nr = np.zeros_like(a)
        Theta_rec = np.zeros_like(a)

    # Common processing for both cases
    # Determine Tau (separation angle)
    scalar_tau = (
        S_rec1 * np.conj(DS_1)
        + S_rec2 * np.conj(DS_2)
        + S_rec3 * np.conj(DS_3)
        + S_rec4 * np.conj(DS_4)
    )

    S_rec_norm = np.sqrt(
        np.abs(S_rec1) ** 2
        + np.abs(S_rec2) ** 2
        + np.abs(S_rec3) ** 2
        + np.abs(S_rec4) ** 2
        + eps  # Avoid division by zero
    )
    DS_norm = np.sqrt(
        np.abs(DS_1) ** 2 + np.abs(DS_2) ** 2 + np.abs(DS_3) ** 2 + np.abs(DS_4) ** 2
    )

    # Avoid division by zero
    denom = S_rec_norm * DS_norm
    tau_arg = np.abs(scalar_tau) / (denom + eps)  # Avoid division by zero 
    tau_arg = np.clip(tau_arg, 0, 1)
    Tau = np.arccos(tau_arg)

    # Determine Psi_D (diagonalization angle)
    Psi_1 = -x / 4

    # Test three candidate angles
    Psi_candidates = [Psi_1, Psi_1 + np.pi / 2, Psi_1 - np.pi / 2]

    # Initialize Psi_D
    Psi_D = np.zeros_like(Psi_1)

    for Psi_cand in Psi_candidates:
        # Check if angle is in valid range [-π/2, π/2]
        valid_range = (Psi_cand > -np.pi / 2) & (Psi_cand <= np.pi / 2)

        if np.any(valid_range):
            # Compute A1_1 and A1_4 for this candidate
            A1_1 = (
                (np.cos(Psi_cand) ** 2) * DS_rec1
                - (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec2
                - (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec3
                + (np.sin(Psi_cand) ** 2) * DS_rec4
            )

            A1_4 = (
                (np.sin(Psi_cand) ** 2) * DS_rec1
                + (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec2
                + (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec3
                + (np.cos(Psi_cand) ** 2) * DS_rec4
            )

            # Select where A1_1 >= A1_4 and angle is in valid range and Psi_D not set
            select_mask = valid_range & (np.abs(A1_1) >= np.abs(A1_4)) & (Psi_D == 0)
            Psi_D[select_mask] = Psi_cand[select_mask]

    # Apply final Psi_D correction based on diagonal condition
    A1_1_final = (
        (np.cos(Psi_D) ** 2) * DS_rec1
        - (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec2
        - (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec3
        + (np.sin(Psi_D) ** 2) * DS_rec4
    )

    A1_4_final = (
        (np.sin(Psi_D) ** 2) * DS_rec1
        + (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec2
        + (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec3
        + (np.cos(Psi_D) ** 2) * DS_rec4
    )

    # Additional Psi_D adjustments based on diagonal conditions
    I_a = A1_1_final == A1_4_final
    I_b = A1_1_final == -A1_4_final

    mask1 = (Psi_D > np.pi / 4) & (I_a | I_b)
    mask2 = (Psi_D > -np.pi / 4) & (Psi_D <= np.pi / 4) & (~mask1) & (I_a | I_b)
    mask3 = (Psi_D <= -np.pi / 4) & (~mask1) & (~mask2) & (I_a | I_b)

    Psi_D[mask1] = Psi_D[mask1] - np.pi / 2
    # mask2 keeps original Psi_D
    Psi_D[mask3] = Psi_D[mask3] + np.pi / 2

    # Always return 13 parameters for test compatibility
    return [
        S_max1,
        S_max2,
        S_max3,
        S_max4,
        S_min1,
        S_min2,
        S_min3,
        S_min4,
        S_nr,
        a,
        Tau,
        Theta_rec,
        Psi_D,
    ]


def cameron(
    fullsamples: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Optimized Cameron classification.

    Args:
        S_max1..S_max4, S_min1..S_min4, S_nr, a, Tau, Theta_rec, Psi_D: Parameter arrays from cameron_transform

    Returns:
        classification map (H, W) with integer labels 1-11
    """
    # validate input
    if fullsamples.ndim != 3 or fullsamples.shape[0] not in (3, 4):
        raise ValueError(
            f"Expected fullsamples shape (3,H,W) or (4,H,W), got {fullsamples.shape}"
        )

    # compute decomposition parameters
    S_max1, S_max2, S_max3, S_max4, S_min1, S_min2, S_min3, S_min4, S_nr, a, Tau, Theta_rec, Psi_D = cameron_transform(fullsamples)

    # Precompute trigonometric terms
    cos_tr = np.cos(Theta_rec)
    sin_tr = np.sin(Theta_rec)
    cos_ta = np.cos(Tau)
    sin_ta = np.sin(Tau)

    # Compute scattering matrix elements S1..S4
    s1 = a * cos_tr * (cos_ta * S_max1 + sin_ta * S_min1)
    s2 = a * cos_tr * (cos_ta * S_max2 + sin_ta * S_min2) - (
        a * sin_tr * S_nr / np.sqrt(2)
    )
    s3 = a * cos_tr * (cos_ta * S_max3 + sin_ta * S_min3) + (
        a * sin_tr * S_nr / np.sqrt(2)
    )
    s4 = a * cos_tr * (cos_ta * S_max4 + sin_ta * S_min4)

    H, W = a.shape
    cls = np.zeros((H, W), dtype=int)

    # 1) Non-reciprocal
    nr = Theta_rec > np.pi / 4
    cls[nr] = 1

    # 2) Asymmetric & helix
    hel = (~nr) & (Tau > np.pi / 8)
    left = 0.5 * (s1 - s4 - 1j * (s2 + s3))
    right = 0.5 * (s1 - s4 + 1j * (s2 + s3))
    th_l = np.arccos(np.clip(np.abs(left) / (a + eps), 0, 1))
    th_r = np.arccos(np.clip(np.abs(right) / (a + eps), 0, 1))

    asym = hel & (th_l > np.pi / 4) & (th_r > np.pi / 4)
    cls[asym] = 2

    lh = hel & ~asym & (th_l >= th_r)
    cls[lh] = 3
    rh = hel & ~asym & (th_l < th_r)
    cls[rh] = 4

    # 3) Symmetric vs canonical targets
    sym = (~nr) & (Tau <= np.pi / 8)
    # compute canonical scalars
    tri = (s1 + s4) / np.sqrt(2)
    dihedral = (s1 - s4) / np.sqrt(2)
    dipole = s1
    cylinder = (2 * s1 + s4) / np.sqrt(5)
    narrow = (2 * s1 - s4) / np.sqrt(5)
    quarter = (s1 - 1j * s4) / np.sqrt(2)

    # angles to each target
    angles = np.stack(
        [
            np.arccos(np.clip(np.abs(tri) / (a + eps), 0, 1)),
            np.arccos(np.clip(np.abs(dihedral) / (a + eps), 0, 1)),
            np.arccos(np.clip(np.abs(dipole) / (a + eps), 0, 1)),
            np.arccos(np.clip(np.abs(cylinder) / (a + eps), 0, 1)),
            np.arccos(np.clip(np.abs(narrow) / (a + eps), 0, 1)),
            np.arccos(np.clip(np.abs(quarter) / (a + eps), 0, 1)),
        ],
        axis=0,
    )  # shape (6, H, W)

    min_ang = angles.min(axis=0)
    # Symmetric
    cls[sym & (min_ang > np.pi / 4)] = 5

    # Assign canonical classes 6-11
    idx = angles.argmin(axis=0)
    for k in range(6):
        mask_k = sym & (idx == k) & (min_ang <= np.pi / 4)
        cls[mask_k] = 6 + k
    return cls

def rss(x: Union[np.ndarray, torch.Tensor], dim: int = 0):# -> Any:
    # If multi-channel (Coils), apply RSS: sqrt(sum(|x|^2))
    if x.shape[0] > 1:
        if isinstance(x, np.ndarray):
            return np.sqrt(np.sum(np.abs(x)**2, axis=dim))
        elif isinstance(x, torch.Tensor):
            return torch.sqrt(x.abs().pow(2).sum(dim=dim))
        else:
            raise TypeError("Input must be a numpy array or a torch tensor")
    # If single-channel (already combined), just magnitude
    else:
        if isinstance(x, np.ndarray):
            return np.abs(x).squeeze()
        elif isinstance(x, torch.Tensor):
            return x.abs().squeeze()
        else:
            raise TypeError("Input must be a numpy array or a torch tensor")