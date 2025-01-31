import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import sklearn.metrics as skm
from sklearn.preprocessing import StandardScaler
from skimage import exposure
from scipy.linalg import eigh
from .utils import compute_metrics, apply_kmeans
from .models.projection import binomial_expansion
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
import os
import wandb
from collections.abc import MutableMapping
import plotly.graph_objects as go
import umap
import plotly.colors
import plotly.express as px

MIN_VALUE = 0.02
MAX_VALUE = 40


def pauli_transform(sar_img: np.ndarray) -> np.ndarray:
    """
    Perform Pauli decomposition on the SAR image.
    """
    s_hh = sar_img[:, :, 0]
    s_hv = sar_img[:, :, 1]
    s_vv = sar_img[:, :, 2]
    return (1 / np.sqrt(2)) * np.stack(
        (s_hh - s_vv, 2 * s_hv, s_hh + s_vv), dtype=np.complex64
    )


def cameron_transform(sar_img: np.ndarray) -> tuple:
    """
    Perform Cameron decomposition on the SAR image.
    """
    s_hh = sar_img[:, :, 0]
    s_hv = sar_img[:, :, 1]
    s_vv = sar_img[:, :, 2]
    s_vh = s_hv

    a = np.sqrt(
        s_hh * np.conj(s_hh)
        + s_hv * np.conj(s_hv)
        + s_vh * np.conj(s_vh)
        + s_vv * np.conj(s_vv)
    )

    alpha, beta, gamma, delta = (
        1 / np.sqrt(2) * (s_hh + s_vv),
        1 / np.sqrt(2) * (s_hh - s_vv),
        1 / np.sqrt(2) * (s_hv + s_vh),
        1 / np.sqrt(2) * (s_vh - s_hv),
    )

    sin_x, cos_x = compute_sin_cos_x(beta, gamma)
    x = compute_x(sin_x, cos_x)

    ds_1, ds_2, ds_3, ds_4 = compute_ds(alpha, s_hh, s_hv, s_vh, s_vv, x)
    s_max1, s_max2, s_max3, s_max4 = normalize_components(ds_1, ds_2, ds_3, ds_4)

    s_rec1, s_rec2, s_rec3, s_rec4 = (
        s_hh,
        0.5 * (s_hv + s_vh),
        0.5 * (s_hv + s_vh),
        s_vv,
    )
    ds_rec1, ds_rec2, ds_rec3, ds_rec4 = compute_ds(
        alpha, s_rec1, s_rec2, s_rec3, s_rec4, x
    )
    s_min1, s_min2, s_min3, s_min4 = normalize_components(
        s_rec1 - ds_rec1, s_rec2 - ds_rec2, s_rec3 - ds_rec3, s_rec4 - ds_rec4
    )

    s_nr = delta / np.abs(delta)

    theta_rec = np.arccos(
        np.sqrt(
            s_rec1 * np.conj(s_rec1)
            + s_rec2 * np.conj(s_rec2)
            + s_rec3 * np.conj(s_rec3)
            + s_rec4 * np.conj(s_rec4)
        )
        / a
    )

    tau = compute_tau(s_rec1, s_rec2, s_rec3, s_rec4, ds_1, ds_2, ds_3, ds_4)
    psi_d = compute_psi_d(x, ds_rec1, ds_rec2, ds_rec3, ds_rec4)

    return (
        s_max1,
        s_max2,
        s_max3,
        s_max4,
        s_min1,
        s_min2,
        s_min3,
        s_min4,
        s_nr,
        a,
        tau,
        theta_rec,
        psi_d,
    )


def compute_sin_cos_x(beta, gamma):
    """
    Compute sin(x) and cos(x) values for Cameron decomposition.
    """
    sin_x = (beta * np.conj(gamma) + np.conj(beta) * gamma) / np.sqrt(
        (beta * np.conj(gamma) + np.conj(beta) * gamma) ** 2
        + (np.abs(beta) ** 2 - np.abs(gamma) ** 2) ** 2
    )
    cos_x = (np.abs(beta) ** 2 - np.abs(gamma) ** 2) / np.sqrt(
        (beta * np.conj(gamma) + np.conj(beta) * gamma) ** 2
        + (np.abs(beta) ** 2 - np.abs(gamma) ** 2) ** 2
    )
    return sin_x, cos_x


def compute_x(sin_x, cos_x):
    """
    Compute x value for Cameron decomposition.
    """
    return (
        np.arccos(cos_x) * (sin_x >= 0)
        + np.arcsin(sin_x) * ((sin_x < 0) & (cos_x >= 0))
        + (-np.arcsin(sin_x) - np.pi) * ((sin_x < 0) & (cos_x < 0))
    ) * (
        (np.abs(sin_x) ** 2 - np.abs(cos_x) ** 2 != 0)
        | (sin_x * np.conj(cos_x) + np.conj(sin_x) * cos_x != 0)
    )


def compute_ds(alpha, s_hh, s_hv, s_vh, s_vv, x):
    """
    Compute DS components for Cameron decomposition.
    """
    scalar = (
        1
        / np.sqrt(2)
        * (
            s_hh * np.cos(x / 2)
            + s_hv * np.sin(x / 2)
            + s_vh * np.sin(x / 2)
            - s_vv * np.cos(x / 2)
        )
    )
    ds_1 = 1 / np.sqrt(2) * (alpha + np.cos(x / 2) * scalar)
    ds_2 = 1 / np.sqrt(2) * np.sin(x / 2) * scalar
    ds_3 = 1 / np.sqrt(2) * np.sin(x / 2) * scalar
    ds_4 = 1 / np.sqrt(2) * (alpha - (np.cos(x / 2) * scalar))
    return ds_1, ds_2, ds_3, ds_4


def normalize_components(*components):
    """
    Normalize the DS components.
    """
    s_max = np.sqrt(sum([comp * np.conj(comp) for comp in components]))
    return [comp / s_max for comp in components]


def compute_tau(s_rec1, s_rec2, s_rec3, s_rec4, ds_1, ds_2, ds_3, ds_4):
    """
    Compute Tau for Cameron decomposition.
    """
    scalar_tau = (
        s_rec1 * np.conj(ds_1)
        + s_rec2 * np.conj(ds_2)
        + s_rec3 * np.conj(ds_3)
        + s_rec4 * np.conj(ds_4)
    )
    s_max_tau = np.sqrt(
        s_rec1 * np.conj(s_rec1)
        + s_rec2 * np.conj(s_rec2)
        + s_rec3 * np.conj(s_rec3)
        + s_rec4 * np.conj(s_rec4)
    )
    s_maxx_tau = np.sqrt(
        ds_1 * np.conj(ds_1)
        + ds_2 * np.conj(ds_2)
        + ds_3 * np.conj(ds_3)
        + ds_4 * np.conj(ds_4)
    )
    return np.arccos(np.abs(scalar_tau / (s_max_tau * s_maxx_tau)))


def compute_psi_d(x, ds_rec1, ds_rec2, ds_rec3, ds_rec4):
    """
    Compute Psi_D for Cameron decomposition.
    """
    psi_1 = -1 / 4 * x
    psi_11 = psi_1
    psi_12 = psi_1 + np.pi / 2
    psi_13 = psi_1 - np.pi / 2

    def compute_a_components(psi):
        a_1 = (
            (np.cos(psi) ** 2) * ds_rec1
            - (np.cos(psi) * np.sin(psi)) * ds_rec2
            - (np.cos(psi) * np.sin(psi)) * ds_rec3
            + (np.sin(psi) ** 2) * ds_rec4
        )
        a_4 = (
            (np.sin(psi) ** 2) * ds_rec1
            + (np.cos(psi) * np.sin(psi)) * ds_rec2
            + (np.cos(psi) * np.sin(psi)) * ds_rec3
            + (np.cos(psi) ** 2) * ds_rec4
        )
        return np.abs(a_1), np.abs(a_4)

    a1_1, a1_4 = compute_a_components(psi_11)
    a2_1, a2_4 = compute_a_components(psi_12)
    a3_1, a3_4 = compute_a_components(psi_13)

    psi_0 = psi_11 * ((psi_11 > -np.pi / 2) & (psi_11 <= np.pi / 2) & (a1_1 >= a1_4))
    psi_0 = psi_0 + psi_12 * (
        (psi_12 > -np.pi / 2) & (psi_12 <= np.pi / 2) & (a2_1 >= a2_4) & (psi_0 == 0)
    )
    psi_0 = psi_0 + psi_13 * (
        (psi_13 > -np.pi / 2) & (psi_13 <= np.pi / 2) & (a3_1 >= a3_4) & (psi_0 == 0)
    )

    a1_1, a1_4 = compute_a_components(psi_0)
    i_a = a1_1 == a1_4
    i_b = a1_1 == -a1_4

    psi_d = (psi_0 - np.pi / 2) * ((psi_0 > np.pi / 4) & (i_a | i_b))
    psi_d = psi_d + psi_0 * (
        ((psi_0 > -np.pi / 4) & (psi_0 <= np.pi / 4) & (psi_d == 0)) & (i_a | i_b)
    )
    psi_d = psi_d + (psi_0 + np.pi / 2) * (
        ((psi_0 <= -np.pi / 4) & (psi_d == 0)) & (i_a | i_b)
    )
    psi_d = psi_d + psi_0 * ((i_a == 0) & (i_b == 0))

    return psi_d


def cameron_classification(
    s_max1: np.ndarray,
    s_max2: np.ndarray,
    s_max3: np.ndarray,
    s_max4: np.ndarray,
    s_min1: np.ndarray,
    s_min2: np.ndarray,
    s_min3: np.ndarray,
    s_min4: np.ndarray,
    s_nr: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    theta_rec: np.ndarray,
    psi_d: np.ndarray,
) -> np.ndarray:
    """
    Perform classification using Cameron decomposition.
    """
    a1 = (
        (np.cos(psi_d) ** 2) * s_max1
        - (np.cos(psi_d) * np.sin(psi_d)) * (s_max2 + s_max3)
        + (np.sin(psi_d) ** 2) * s_max4
    )
    a4 = (
        (np.sin(psi_d) ** 2) * s_max1
        + (np.cos(psi_d) * np.sin(psi_d)) * (s_max2 + s_max3)
        + (np.cos(psi_d) ** 2) * s_max4
    )
    z = a4 / a1

    classe = np.zeros(theta_rec.shape)

    for i in range(theta_rec.shape[0]):
        for j in range(theta_rec.shape[1]):
            if theta_rec[i, j] > np.pi / 4:
                classe[i, j] = 1
            elif theta_rec[i, j] <= np.pi / 4 and tau[i, j] > np.pi / 8:
                s1, s2, s3, s4 = compute_s_values(
                    a,
                    i,
                    j,
                    theta_rec,
                    tau,
                    s_max1,
                    s_max2,
                    s_max3,
                    s_max4,
                    s_min1,
                    s_min2,
                    s_min3,
                    s_min4,
                    s_nr,
                )
                scalarleft = 0.5 * (s1 - s4 - 1j * (s2 + s3))
                scalarright = 0.5 * (s1 - s4 + 1j * (s2 + s3))

                theta_tleft = np.arccos(abs(scalarleft / a[i, j]))
                theta_tright = np.arccos(abs(scalarright / a[i, j]))

                if theta_tleft > np.pi / 4 and theta_tright > np.pi / 4:
                    classe[i, j] = 2
                else:
                    classifieur = int(theta_tleft >= theta_tright)
                    classe[i, j] = classifieur * 3 + (1 - classifieur) * 4

            elif theta_rec[i, j] <= np.pi / 4 and tau[i, j] <= np.pi / 8:
                classifieur = classify_theta_tau(z, i, j)
                classe[i, j] = (
                    classifieur
                    if classifieur > np.pi / 4
                    else np.argmin(classifieur) + 6
                )

    return classe


def compute_s_values(
    a,
    i,
    j,
    theta_rec,
    tau,
    s_max1,
    s_max2,
    s_max3,
    s_max4,
    s_min1,
    s_min2,
    s_min3,
    s_min4,
    s_nr,
):
    """
    Compute S values for Cameron classification.
    """
    s1 = (
        a[i, j]
        * np.cos(theta_rec[i, j])
        * (np.cos(tau[i, j]) * s_max1[i, j] + np.sin(tau[i, j]) * s_min1[i, j])
    )
    s2 = a[i, j] * np.cos(theta_rec[i, j]) * (
        np.cos(tau[i, j]) * s_max2[i, j] + np.sin(tau[i, j]) * s_min2[i, j]
    ) - a[i, j] * np.sin(theta_rec[i, j]) * s_nr[i, j] / np.sqrt(2)
    s3 = a[i, j] * np.cos(theta_rec[i, j]) * (
        np.cos(tau[i, j]) * s_max3[i, j] + np.sin(tau[i, j]) * s_min3[i, j]
    ) + a[i, j] * np.sin(theta_rec[i, j]) * s_nr[i, j] / np.sqrt(2)
    s4 = (
        a[i, j]
        * np.cos(theta_rec[i, j])
        * (np.cos(tau[i, j]) * s_max4[i, j] + np.sin(tau[i, j]) * s_min4[i, j])
    )
    return s1, s2, s3, s4


def classify_theta_tau(z, i, j):
    """
    Classify based on theta and tau values.
    """
    z_conj = np.conj(z[i, j])
    d_trihedre = compute_d(z_conj, 1, np.sqrt(2 * (1 + np.abs(z[i, j]) ** 2)))
    d_dihedre = compute_d(z_conj, 1, np.sqrt(2 * (1 + np.abs(z[i, j]) ** 2)))
    d_dipole = compute_d(z_conj, 1, np.sqrt((1 + np.abs(z[i, j]) ** 2)))
    d_cylindre = compute_d(z_conj, 1 / 2, np.sqrt(5 / 4 * (1 + np.abs(z[i, j]) ** 2)))
    d_dihedreetroit = compute_d(
        z_conj, -1 / 2, np.sqrt(5 / 4 * (1 + np.abs(z[i, j]) ** 2))
    )
    d_quartonde = compute_d(z_conj, 1j, np.sqrt(2 * (1 + np.abs(z[i, j]) ** 2)))

    d = np.array(
        [d_trihedre, d_dihedre, d_dipole, d_cylindre, d_dihedreetroit, d_quartonde]
    )
    return np.min(d)


def compute_d(z_conj, factor, denominator):
    """
    Compute D value for classification.
    """
    return np.arccos(
        np.maximum(np.abs(1 + factor * z_conj), np.abs(factor + z_conj)) / denominator
    )


def krogager_transform(sar_img: np.ndarray) -> np.ndarray:
    """
    Perform Krogager decomposition on the SAR image.
    """
    s_hh = sar_img[:, :, 0]
    s_hv = sar_img[:, :, 1]
    s_vv = sar_img[:, :, 2]
    s_rr = 1j * s_hv + 0.5 * (s_hh - s_vv)
    s_ll = 1j * s_hv - 0.5 * (s_hh - s_vv)
    s_rl = 1j / 2 * (s_hh + s_vv)

    return np.stack(
        (
            np.minimum(np.abs(s_rr), np.abs(s_ll)),
            np.abs(np.abs(s_rr) - np.abs(s_ll)),
            np.abs(s_rl),
        )
    )


def exp_amplitude_transform(tensor: np.ndarray) -> torch.Tensor:
    """
    Apply exponential amplitude transformation to the tensor.
    """
    tensor = torch.from_numpy(tensor)
    new_tensor = []

    for idx, ch in enumerate(tensor):
        amplitude = torch.abs(ch)
        phase = torch.angle(ch)
        min_val = MIN_VALUE
        max_val = MAX_VALUE

        inv_transformed_amplitude = torch.clip(
            torch.exp(
                (
                    (np.log10(max_val) - np.log10(min_val)) * amplitude
                    + np.log10(min_val)
                )
                * np.log(10)
            ),
            0,
            10**9,
        )

        new_tensor.append(inv_transformed_amplitude * torch.exp(1j * phase))

    return torch.as_tensor(np.stack(new_tensor), dtype=torch.complex64)


def equalize(image: np.ndarray, p2: float = None, p98: float = None) -> tuple:
    """
    Automatically adjust the contrast of the SAR image (intensity or amplitude in dB scale).
    """
    img = np.log10(np.abs(image) + 1e-16)
    if p2 is None or p98 is None:
        p2, p98 = np.percentile(img, (2, 98))
    img_resc = np.round(
        exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1)) * 255
    ).astype(np.uint8)

    return img_resc, (p2, p98)


def angular_distance(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute the angular distance between two phase angles.
    """
    diff = np.angle(image1) - np.angle(image2) + np.pi
    angular_dist = np.mod(diff, 2 * np.pi) - np.pi
    return angular_dist


def plot_phase(image: np.ndarray) -> np.ndarray:
    """
    Plot the phase of a PolSAR image and normalize it to [0, 255].
    """
    phase_image = np.angle(image)
    normalized_phase = (phase_image + np.pi) / (2 * np.pi)
    return np.round(normalized_phase * 255).astype(np.uint8)


def plot_angular_distance(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Plot the phase of a PolSAR image and normalize it to [0, 255].
    """
    ang_distance_image = angular_distance(image1, image2)
    normalized_ang_distance_image = (ang_distance_image + np.pi) / (2 * np.pi)
    return np.round(normalized_ang_distance_image * 255).astype(np.uint8)


def plot_fourier_transform_amplitude_phase(image: np.ndarray) -> tuple:
    """
    Plot the Fourier transform amplitude and phase of the image.
    """
    amplitude_ft_images, phase_ft_vectors = [], []

    for channel in range(image.shape[0]):
        fft_img = np.fft.fftshift(np.fft.fft2(image[channel, :, :]))
        amplitude = np.abs(fft_img)
        phase = np.angle(fft_img)

        amplitude_ft_images.append(np.log(np.abs(amplitude) + 1))
        phase_ft_vectors.append((np.cos(phase), np.sin(phase)))

    return amplitude_ft_images, phase_ft_vectors


def calculate_means_of_classes(
    image_of_stacked_covariances: np.ndarray, classes_h_alpha: np.ndarray
) -> dict:
    """
    Calculate the means of the classes after H-alpha initialization.
    """
    list_of_classes = [1, 2, 4, 5, 6, 7, 8, 9]
    dictionary_of_means = {}

    for k in list_of_classes:
        mask = classes_h_alpha == k
        size_of_mask_1, size_of_mask_2 = mask.shape
        mask = np.reshape(mask, (size_of_mask_1, size_of_mask_2, 1))
        cov_times_mask = image_of_stacked_covariances * mask
        cov_times_mask = np.reshape(
            cov_times_mask,
            (
                cov_times_mask.shape[0] * cov_times_mask.shape[1],
                cov_times_mask.shape[2],
            ),
        )
        mean_of_class = np.mean(cov_times_mask, axis=0)
        dictionary_of_means["mean" + str(k)] = mean_of_class

    return dictionary_of_means


def h_alpha(pauli_radar_image: np.ndarray) -> np.ndarray:
    """
    Perform H-alpha decomposition and initialization of the classes.
    """
    s1, s2, p = pauli_radar_image.shape
    son = 7

    p_vector, alpha_vector = np.zeros(3), np.zeros(3)
    h_alpha = np.zeros((s1 - (son - 1), s2 - (son - 1), 2))
    classes_h_alpha_original = np.zeros((s1 - (son - 1), s2 - (son - 1)), dtype=int)
    covariances_stacked = np.zeros(
        (s1 - (son - 1), s2 - (son - 1), p * p), dtype=complex
    )

    for k in range(s1 - (son - 1)):
        for l in range(s2 - (son - 1)):
            local_data_matrix = np.reshape(
                pauli_radar_image[k : k + son, l : l + son, :], (son**2, p)
            )
            local_covariance = np.dot(
                np.conjugate(local_data_matrix).T, local_data_matrix
            ) / (son**2)
            local_covariance_stacked = np.reshape(local_covariance, (1, 1, p * p))
            covariances_stacked[k, l, :] = local_covariance_stacked
            eigenvalues, eigenvectors = eigh(local_covariance)

            p_vector = eigenvalues / np.sum(eigenvalues)
            alpha_vector = np.arccos(np.abs(eigenvectors[0, :]))

            h, alpha = compute_h_alpha(p_vector, alpha_vector)
            h_alpha[k, l, 0], h_alpha[k, l, 1] = h, alpha

            classes_h_alpha_original[k, l] = assign_h_alpha_class(h, alpha)

    return classes_h_alpha_original


def compute_h_alpha(p_vector: np.ndarray, alpha_vector: np.ndarray) -> tuple:
    """
    Compute H and alpha values for H-alpha decomposition.
    """
    h = -np.dot(p_vector, np.log(p_vector + 1e-5))
    h = 1.0 if h > 1.0 else h
    h = 0 if np.isnan(h) else h
    alpha = np.dot(p_vector, alpha_vector) * (180.0 / np.pi)
    alpha = 90 if alpha > 90 else alpha
    return h, alpha


def assign_h_alpha_class(h: float, alpha: float) -> int:
    """
    Assign class based on H and alpha values.
    """
    if h <= 0.5:
        if alpha <= 42.5:
            return 9
        elif alpha <= 47.5:
            return 8
        else:
            return 7
    elif h <= 0.9:
        if alpha <= 40:
            return 6
        elif alpha <= 50:
            return 5
        else:
            return 4
    else:
        if alpha <= 55:
            return 2
        else:
            return 1


def show_images(
    ground_truth: list,
    predicted: list,
    image_path: str,
    task: str,
    metrics: dict = None,
    last: bool = False,
    test: bool = False,
    number_classes: int = None,
    ignore_index: int = -100,
    test_mask=None,
    config: dict = None,
    wandb_log=None,
    is_mnist=False,
) -> None:
    """
    Display images for ground truth and predictions.
    """
    num_samples = len(ground_truth)
    if task == "reconstruction" and not is_mnist:
        num_channels = ground_truth[0].shape[0]
    else:
        num_channels = None

    ncols = calculate_ncols(task, test, last, num_channels, test_mask, is_mnist)

    fig, axes = plt.subplots(
        num_samples,
        ncols,
        figsize=(5 * ncols, 5 * num_samples),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    res = None

    for i in range(num_samples):  # rework cette partie

        idx = 0
        g_t = np.copy(ground_truth[i])
        pred = np.copy(predicted[i])
        if test_mask is not None:
            test_mask = np.copy(test_mask[i])

        if task == "segmentation":
            res = plot_segmentation_images(
                g_t, pred, axes, i, idx, number_classes, ignore_index, test, test_mask
            )
        elif task == "reconstruction":
            if is_mnist:
                res = plot_reconstruction_mnist_images(g_t, pred, axes, i, idx, test)
            else:
                res = plot_reconstruction_polsar_images(
                    g_t, pred, axes, i, idx, test, last, num_channels
                )
        elif task == "classification":
            res = plot_classification_images(
                g_t, pred, axes, i, idx, number_classes, ignore_index, test
            )

    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    if test:
        metrics.update(res)
        metrics["Task"] = task
        if wandb_log:
            wandb.log(metrics)
            wandb.log({"test_image": [wandb.Image(fig, caption="Test Image")]})
        else:
            log_model_performance(
                hyperparameters=config,
                metrics=metrics,
                excel_file_path=config["logging"]["wandb"]["project"] + ".xlsx",
                task=task,
            )


def calculate_ncols(
    task: str,
    test: bool,
    last: bool,
    num_channels: int,
    test_mask: np.array,
    is_mnist: bool = False,
) -> int:
    """
    Calculate the number of columns for the subplot grid.
    """
    if task == "segmentation":
        return 2 if not test else 3 if test_mask is None else 4
    elif task == "reconstruction":
        if is_mnist:
            return 2
        else:
            ncols = 9 + 3 * test + 4 * last * num_channels
            return ncols
    elif task == "classification":
        return 1 if not test else 2

    return 1


def plot_segmentation_images(
    to_be_visualized: list,
    confusion_matrix: np.ndarray,
    number_classes: int,
    logdir: str,
    wandb_log: bool,
    ignore_index: int = None,
    sets_masks: np.ndarray = None,
) -> None:
    """
    Plots segmentation images with an optional test mask overlay to indicate dataset splits.

    Args:
        to_be_visualized (list): Array of shape (N, 3, H, W), where:
                                       - First channel: Ground truth.
                                       - Second channel: Prediction.
                                       - Third channel: Original image (optional).
        confusion_matrix (np.ndarray): Confusion matrix of shape (number_classes, number_classes).
        number_classes (int): Number of classes for segmentation.
        logdir (str): Directory to save the plot.
        wandb_log (bool): Whether to log the plot to Weights & Biases.
        ignore_index (int, optional): Value in the ground truth to be ignored in the masked prediction.
        sets_masks (np.ndarray, optional): Array of shape (N, H, W) with integer values indicating dataset splits:
                                           - 1: Train
                                           - 2: Validation
                                           - 3: Test
    """
    # Define colormap for segmentation classes
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
    norm = BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # Define colormap for sets masks
    sets_mask_colors = {
        1: "red",  # Train
        2: "green",  # Validation
        3: "blue",  # Test
    }
    sets_mask_cmap = ListedColormap(
        [sets_mask_colors[key] for key in sorted(sets_mask_colors.keys())]
    )
    sets_mask_bounds = np.arange(len(sets_mask_colors) + 1) - 0.5
    sets_mask_norm = BoundaryNorm(sets_mask_bounds, sets_mask_cmap.N)
    sets_mask_patches = [
        mpatches.Patch(color=sets_mask_colors[i], label=f"Set {i}")
        for i in sorted(sets_mask_colors.keys())
    ]

    # Limit number of samples to visualize
    num_samples = to_be_visualized[0].shape[0]
    nrows = num_samples + 1  # +1 for confusion matrix
    ncols = 4 if sets_masks is not None else 3  # Add test mask column if available

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
    )

    # Plot ground truth, predictions, masked predictions, and optionally test masks
    for i in range(num_samples):
        img = to_be_visualized[0][i]
        g_t = to_be_visualized[1][i]
        pred = to_be_visualized[2][i]

        # Mask prediction if ignore_index is provided
        if ignore_index is not None:
            masked_pred = pred.copy()
            masked_pred[g_t == ignore_index] = ignore_index
        else:
            masked_pred = pred

        # Plot ground truth
        g_t = np.squeeze(g_t)
        axes[i][0].imshow(g_t, cmap=cmap, norm=norm, origin="lower")
        axes[i][0].set_title(f"Ground Truth {i+1}")
        axes[i][0].axis("off")

        # Plot prediction
        pred = np.squeeze(pred)
        axes[i][1].imshow(pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][1].set_title(f"Prediction {i+1}")
        axes[i][1].axis("off")

        # Plot masked prediction
        masked_pred = np.squeeze(masked_pred)
        axes[i][2].imshow(masked_pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][2].set_title(f"Masked Prediction {i+1}")
        axes[i][2].axis("off")

        # Plot test mask if available
        if sets_masks is not None:
            axes[i][3].imshow(sets_masks[i], cmap=sets_mask_cmap, norm=sets_mask_norm)
            axes[i][3].set_title(f"Sets Mask {i+1}")
            axes[i][3].axis("off")

    # Plot confusion matrix in the last row
    sns.heatmap(
        confusion_matrix.round(decimals=3),
        annot=True,
        fmt=".2g",
        cmap="Blues",
        ax=axes[-1][0],
        xticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
        yticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
    )
    axes[-1][0].set_xlabel("Predicted Class")
    axes[-1][0].set_ylabel("Ground Truth Class")
    axes[-1][0].set_title("Confusion Matrix")

    # Add legends
    legend_ax = axes[-1][1]
    legend_ax.axis("off")
    legend_ax.legend(handles=patches, loc="center", title="Classes")

    # Add sets mask legend if applicable
    if sets_masks is not None:
        test_mask_legend_ax = axes[-1][2]
        test_mask_legend_ax.axis("off")
        test_mask_legend_ax.legend(
            handles=sets_mask_patches, loc="center", title="Test Masks"
        )
    else:
        axes[-1][2].axis("off")

    # Leave extra columns blank for symmetry
    if ncols == 4:
        axes[-1][3].axis("off")

    # Save the figure
    path = f"{logdir}/segmentation_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Log to Weights & Biases if enabled
    if wandb_log:
        wandb.log(
            {
                "segmentation_images": [
                    wandb.Image(
                        path, caption="Segmentation Images and Confusion Matrix"
                    )
                ]
            }
        )


def plot_classification_images(
    to_be_visualized: list,
    confusion_matrix: np.ndarray,
    number_classes: int,
    logdir: str,
    wandb_log: bool,
) -> None:

    num_samples = len(to_be_visualized)
    nrows = num_samples + 1  # +1 for confusion matrix
    ncols = 2  # To keep images in two columns

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
    )

    # Plot confusion matrix in the first row
    sns.heatmap(
        confusion_matrix.round(decimals=3),
        annot=True,
        fmt=".2g",
        cmap="Blues",
        ax=axes[0][0],
        xticklabels=np.arange(number_classes),
        yticklabels=np.arange(number_classes),
    )
    axes[0][0].set_xlabel("Predicted Class")
    axes[0][0].set_ylabel("Ground Truth Class")
    axes[0][0].set_title("Confusion Matrix")

    # Leave the second column in the first row blank
    axes[0][1].axis("off")

    # Plot classification images
    for i in range(num_samples):
        row_idx = i + 1
        axes[row_idx][0].imshow(to_be_visualized[i][0])
        axes[row_idx][0].set_title(f"Image {i+1} (Sample)")
        axes[row_idx][0].axis("off")

        # Plot the label and the prediction
        axes[row_idx][1].text(
            0.5,
            0.5,
            f"Label: {to_be_visualized[i][1]}\nPrediction: {to_be_visualized[i][2]}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[row_idx][1].transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )
        axes[row_idx][1].axis("off")

    # Save the figure
    path = f"{logdir}/classification_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Log to Weights & Biases if enabled
    if wandb_log:
        wandb.log(
            {
                "classification_images": [
                    wandb.Image(
                        path, caption="Classification Images and Confusion Matrix"
                    )
                ]
            }
        )


def plot_synchrony_images(
    to_be_visualized: list,
    logdir: str,
    wandb_log: bool,
):

    num_samples = len(to_be_visualized)  # Number of samples
    ncols = 3  # Ground truth, Prediction, and Original Image
    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=ncols,
        figsize=(5 * ncols, 5 * num_samples),
        constrained_layout=True,
    )

    axes = np.atleast_2d(axes)  # Ensure axes are always 2D for consistent indexing

    for i in range(num_samples):
        # Extract ground truth, prediction, and image
        g_t = to_be_visualized[i][0]
        pred = to_be_visualized[i][1]
        img = to_be_visualized[i][2]

        # Plot Ground Truth
        axes[i][0].imshow(g_t, cmap="viridis")
        axes[i][0].set_title(f"Ground Truth {i+1}")
        axes[i][0].axis("off")

        # Plot Prediction
        axes[i][1].imshow(pred, cmap="viridis")
        axes[i][1].set_title(f"Prediction {i+1}")
        axes[i][1].axis("off")

        # Plot Original Image
        axes[i][2].imshow(img)
        axes[i][2].set_title(f"Original Image {i+1}")
        axes[i][2].axis("off")

    # Save the figure
    path = f"{logdir}/synchrony_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Log to Weights & Biases if enabled
    if wandb_log:
        wandb.log({"synchrony_images": [wandb.Image(path, caption="Synchrony Images")]})


def plot_reconstruction_polsar_images(
    to_be_visualized: list,
    num_channels: int,
    logdir: str,
    wandb_log: bool,
) -> None:
    """
    Plot reconstruction images for PolSAR data, including amplitude, angular distance,
    histograms, H-alpha decomposition, and confusion matrix.

    Args:
        to_be_visualized (list): List of samples with ground truth and predicted images.
        num_channels (int): Number of channels in the PolSAR images.
        confusion_matrix (np.ndarray): Confusion matrix for H-alpha classification.
    """
    num_samples = to_be_visualized.shape[0]  # Number of samples
    ncols = 12  # Number of plots per sample
    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=ncols,
        figsize=(5 * ncols, 5 * num_samples),
        constrained_layout=True,
    )

    axes = np.atleast_2d(axes)  # Ensure axes are always 2D for consistent indexing

    # Class colors for H-alpha visualization
    class_colors = {
        1: "green",
        2: "yellow",
        4: "blue",
        5: "pink",
        6: "purple",
        7: "red",
        8: "brown",
        9: "gray",
    }
    cmap = ListedColormap(list(class_colors.values()))
    bounds = list(class_colors.keys())
    norm = BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}") for i in class_colors
    ]

    for i in range(num_samples):
        idx = 0

        g_t = to_be_visualized[i, 0]
        pred = to_be_visualized[i, 1]

        # Amplitude images (Pauli and Krogager basis)
        img_ground_truth = exp_amplitude_transform(g_t).numpy().transpose(1, 2, 0)
        img_predicted = exp_amplitude_transform(pred).numpy().transpose(1, 2, 0)

        pauli_img_ground_truth = pauli_transform(img_ground_truth).transpose(1, 2, 0)
        pauli_img_predicted = pauli_transform(img_predicted).transpose(1, 2, 0)

        krogager_img_ground_truth = krogager_transform(img_ground_truth).transpose(
            1, 2, 0
        )
        krogager_img_predicted = krogager_transform(img_predicted).transpose(1, 2, 0)

        # Equalized amplitude images
        eq_img_ground_truth, (p2, p98) = equalize(pauli_img_ground_truth)
        axes[i][idx].imshow(eq_img_ground_truth, origin="lower")
        axes[i][idx].set_title(f"Amplitude GT Pauli {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        eq_img_predicted, _ = equalize(pauli_img_predicted, p2=p2, p98=p98)
        axes[i][idx].imshow(eq_img_predicted, origin="lower")
        axes[i][idx].set_title(f"Amplitude Pred Pauli {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        eq_img_ground_truth, (p2, p98) = equalize(krogager_img_ground_truth)
        axes[i][idx].imshow(eq_img_ground_truth, origin="lower")
        axes[i][idx].set_title(f"Amplitude GT Krogager {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        eq_img_predicted, _ = equalize(krogager_img_predicted, p2=p2, p98=p98)
        axes[i][idx].imshow(eq_img_predicted, origin="lower")
        axes[i][idx].set_title(f"Amplitude Pred Krogager {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        # Angular distance per channel
        channels = ["HH", "HV", "VV"]
        for ch in range(num_channels):
            angular_distance_img = plot_angular_distance(
                img_ground_truth[:, :, ch], img_predicted[:, :, ch]
            )
            axes[i][idx].imshow(angular_distance_img, cmap="hsv", origin="lower")
            axes[i][idx].set_title(f"Angular Dist {channels[ch]} {i + 1}")
            axes[i][idx].axis("off")
            idx += 1

        # Histograms of differences
        mse_values = (np.abs(img_ground_truth) - np.abs(img_predicted)).flatten()
        q5, q95 = np.percentile(mse_values, [5, 95])
        filtered_data = mse_values[(mse_values > q5) & (mse_values < q95)]

        axes[i][idx].hist(filtered_data, bins=100, alpha=0.75)
        axes[i][idx].set_title(f"Amplitude Diff Hist {i + 1}")
        axes[i][idx].set_xlabel("Amplitude Diff")
        axes[i][idx].set_ylabel("Frequency")
        idx += 1

        angular_distance_hist = angular_distance(
            img_ground_truth, img_predicted
        ).flatten()
        axes[i][idx].hist(angular_distance_hist, bins=100, alpha=0.75)
        axes[i][idx].set_title(f"Angular Dist Hist {i + 1}")
        axes[i][idx].set_xlabel("Angular Distance (rad)")
        axes[i][idx].set_ylabel("Frequency")
        idx += 1

        # H-alpha images
        h_alpha_gt = h_alpha(pauli_img_ground_truth)
        h_alpha_pred = h_alpha(pauli_img_predicted)

        axes[i][idx].imshow(h_alpha_gt, origin="lower", cmap=cmap, norm=norm)
        axes[i][idx].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[i][idx].set_title(f"H-alpha GT {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        axes[i][idx].imshow(h_alpha_pred, origin="lower", cmap=cmap, norm=norm)
        axes[i][idx].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[i][idx].set_title(f"H-alpha Pred {i + 1}")
        axes[i][idx].axis("off")
        idx += 1

        confusion_matrix = skm.confusion_matrix(h_alpha_gt, h_alpha_pred)
        # Confusion matrix
        sns.heatmap(
            confusion_matrix.round(decimals=3),
            annot=True,
            fmt=".2g",
            cmap="Blues",
            ax=axes[i][idx],
            xticklabels=list(class_colors.keys()),
            yticklabels=list(class_colors.keys()),
        )
        axes[i][idx].set_xlabel("Predicted")
        axes[i][idx].set_ylabel("Original")
        axes[i][idx].set_title("Confusion Matrix")
        idx += 1

    path = f"{logdir}/reconstruction_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    if wandb_log:
        wandb.log(
            {
                "reconstruction_images": [
                    wandb.Image(
                        f"{logdir}/reconstruction_images.png",
                        caption="Reconstruction Images",
                    )
                ]
            }
        )


def plot_projection_interactive(
    projection, path: str, order: int, wandb_log, range_values, device
):
    # Create a grid of x and y values
    x = torch.linspace(
        range_values["real_min"], range_values["real_max"], 100, dtype=torch.float64
    )
    y = torch.linspace(
        range_values["imag_min"], range_values["imag_max"], 100, dtype=torch.float64
    )

    # Create mesh grid for plotting surface
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Compute amplitude and phase
    R = torch.sqrt(X**2 + Y**2)
    Theta = torch.atan2(Y, X)

    Z = []

    projection = projection.to(device)

    # Compute the projection for each point on the grid
    for i in range(len(x)):
        z_row = []
        for j in range(len(y)):
            x_val = X[i, j]
            y_val = Y[i, j]

            combinations = binomial_expansion(x_val, y_val, order)
            tags = list(combinations.keys())

            # Concatenate the real and imaginary parts
            powers = torch.stack(list(combinations.values())).to(device)

            # Apply the projection function
            z_val = projection(powers).cpu().item()
            z_row.append(z_val)

        Z.append(z_row)

    # Extract LaTeX labels (keys) and scalar values (values)
    latex_labels = tags
    scalar_values = projection.weight[0].detach().cpu().numpy()

    # Plotting the histogram
    fig_hist = plt.figure(figsize=(8, 6))
    plt.bar(latex_labels, scalar_values)

    # Set labels
    plt.xlabel("Combinations")
    plt.ylabel("Values")

    # Enable LaTeX for rendering labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path_hist = path / "projection_coeffs.png"
    plt.savefig(path_hist)
    plt.close()

    if wandb_log:
        wandb.log(
            {
                "histogram_coefficients_poly_function": [
                    wandb.Image(
                        str(path_hist), caption="Histogram Coefficients Poly Function"
                    )
                ]
            }
        )

    # Convert to numpy arrays
    R_np = R.numpy()
    Theta_np = Theta.numpy()
    Z_np = torch.tensor(Z).numpy()

    # Create a Plotly surface plot in polar coordinates
    fig = go.Figure(data=[go.Surface(z=Z_np, x=R_np, y=Theta_np)])

    # Add labels and title
    fig.update_layout(
        title="Projection of Z as a function of Amplitude (R) and Phase (Theta)",
        scene=dict(
            xaxis_title="Amplitude (R)",
            yaxis_title="Phase (Theta)",
            zaxis_title="Z axis (Projected)",
        ),
    )

    path_html_polar = path / "projection_polar.html"

    # Save the plot as an interactive HTML file
    fig.write_html(path_html_polar)

    # Convert to numpy arrays
    X_np = X.numpy()
    Y_np = Y.numpy()

    # Create a Plotly surface plot
    fig = go.Figure(data=[go.Surface(z=Z_np, x=X_np, y=Y_np)])

    # Add labels and title
    fig.update_layout(
        title="Projection of Z as a function of X and Y",
        scene=dict(
            xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Z axis (Projected)"
        ),
    )

    path_html_parts = path / "projection_parts.html"

    # Save the plot as an interactive HTML file
    fig.write_html(path_html_parts)

    if wandb_log:
        wandb.log({f"projection_polar": wandb.Html(str(path_html_polar))})
        wandb.log({f"projection_parts": wandb.Html(str(path_html_parts))})


def plot_latent_features(
    latent_features,
    labels,
    path,
    wandb_log,
    ignore_index=-100,
    n_components=2,
    sample_size=5000,
):
    """
    Processes latent features and labels, determines if labels are 1D or 2D, handles dual-real-valued tensors,
    and visualizes the latent space.

    Parameters:
    - latent_features_list: List of latent feature arrays collected during inference.
    - labels_list: List of label arrays corresponding to the latent features.
    - n_components: Number of components for dimensionality reduction (default=2).
    - sample_size: Number of samples/pixels to use for visualization (default=100000).
    - path: Path object where the HTML file will be saved.
    - wandb_log: Boolean indicating whether to log the visualization to Weights & Biases.
    - ignore_index: Label value to ignore (default=-100).

    Returns:
    - None (saves an interactive HTML plot of the latent space visualization).
    """
    # Concatenate latent features
    # latent_features = np.concatenate(latent_features_list, axis=0)

    # Handle dual-real-valued tensors by merging the last dimension into channels
    if latent_features.ndim == 5 and latent_features.shape[-1] == 2:
        # Merge the last dimension (real and imaginary) into the channel dimension
        batch_size, channels, height, width, dual = latent_features.shape
        latent_features = latent_features.reshape(
            batch_size, channels * dual, height, width
        )
        print(
            f"Latent_features reshaped to merge real and imaginary parts: {latent_features.shape}"
        )
    else:
        print("No dual-real-valued dimension detected or already handled.")

    # Concatenate and squeeze labels
    # labels = np.concatenate(labels_list, axis=0)
    # labels = np.squeeze(labels)

    # Determine if labels are 1D or 2D
    labels_sample = labels[0]

    if labels_sample.ndim == 1:
        # Case 1: Labels are 1D (classification labels)
        print("Labels are 1D (classification labels).")
        # ground_truth is the concatenated labels
        ground_truth = labels
        print("Shape of ground_truth after concatenation:", ground_truth.shape)

        # Determine number of classes
        unique_labels = np.unique(ground_truth)
        number_classes = len(unique_labels)
        print(f"Number of classes: {number_classes}")

        # Define class_colors based on number_classes
        class_colors_dict = {
            8: {
                0: "black",
                1: "purple",
                2: "blue",
                3: "green",
                4: "red",
                5: "cyan",
                6: "yellow",
                7: "orange",
            }
        }
        class_colors = class_colors_dict.get(number_classes, {})
        if not class_colors:
            print(
                f"No predefined color mapping for {number_classes} classes. Using default colors."
            )
            # Assign default colors if mapping is not defined
            unique_labels_sorted = sorted(unique_labels)
            default_colors = px.colors.qualitative.Plotly
            if number_classes > len(default_colors):
                # Repeat colors if not enough
                default_colors *= number_classes // len(default_colors) + 1
            class_colors = {
                label: color
                for label, color in zip(
                    unique_labels_sorted, default_colors[:number_classes]
                )
            }
        else:
            print(f"Using predefined color mapping for {number_classes} classes.")

        # Check that all labels have a color defined
        missing_colors = set(unique_labels) - set(class_colors.keys())
        if missing_colors:
            raise ValueError(f"Missing colors for labels: {missing_colors}")

        # Process latent features
        latent_features_processed = process_latent_features(latent_features)

        # Proceed with visualization
        visualize_latent_space(
            latent_features_processed,
            ground_truth,
            n_components,
            path,
            wandb_log,
            class_colors,
        )

    elif labels_sample.ndim == 2:
        # Case 2: Labels are 2D (segmentation masks)
        print("Labels are 2D (segmentation masks).")
        # Process pixel-wise features and labels
        pixel_features, pixel_labels = process_pixel_wise_features(
            latent_features, labels, sample_size
        )

        # Discard unclassified pixels (e.g., label value == 0)
        classified_indices = pixel_labels != ignore_index
        pixel_features = pixel_features[classified_indices]
        pixel_labels = pixel_labels[classified_indices]
        print(f"Number of classified pixels: {pixel_features.shape[0]}")

        # Determine number of classes
        unique_labels = np.unique(pixel_labels)
        number_classes = len(unique_labels)
        print(f"Number of classes: {number_classes}")

        # Define class_colors based on number_classes
        class_colors_dict = {
            6: {
                1: "purple",
                2: "blue",
                3: "green",
                4: "red",
                5: "cyan",
                6: "yellow",
            },
            4: {
                1: "green",
                2: "brown",
                3: "blue",
                4: "yellow",
            },
            7: {
                1: "green",
                2: "cyan",
                3: "red",
                4: "purple",
                5: "orange",
                6: "yellow",
                7: "blue",
            },
        }
        class_colors = class_colors_dict.get(number_classes, {})
        if not class_colors:
            print(
                f"No predefined color mapping for {number_classes} classes. Using default colors."
            )
            # Assign default colors if mapping is not defined
            unique_labels_sorted = sorted(unique_labels)
            default_colors = px.colors.qualitative.Plotly
            if number_classes > len(default_colors):
                # Repeat colors if not enough
                default_colors *= number_classes // len(default_colors) + 1
            class_colors = {
                label: color
                for label, color in zip(
                    unique_labels_sorted, default_colors[:number_classes]
                )
            }
        else:
            print(f"Using predefined color mapping for {number_classes} classes.")

        # Check that all labels have a color defined
        missing_colors = set(unique_labels) - set(class_colors.keys())
        if missing_colors:
            raise ValueError(f"Missing colors for labels: {missing_colors}")

        # Proceed with visualization
        visualize_latent_space(
            pixel_features, pixel_labels, n_components, path, wandb_log, class_colors
        )

    else:
        raise ValueError("Labels have unsupported number of dimensions.")


def process_latent_features(latent_features):
    """
    Processes latent features by reshaping and normalizing.

    Parameters:
    - latent_features: NumPy array of latent features.

    Returns:
    - processed_features: Normalized features.
    """
    print("Shape of latent_features before processing:", latent_features.shape)

    # Ensure latent_features has at least 2 dimensions
    if latent_features.ndim == 1:
        # Add sample dimension
        latent_features = np.expand_dims(latent_features, axis=0)
        print(f"Added sample dimension: {latent_features.shape}")
    elif latent_features.ndim == 3:
        # If dimensions are (channels, height, width), add batch dimension
        latent_features = np.expand_dims(latent_features, axis=0)
        print(f"Added batch dimension: {latent_features.shape}")
    elif latent_features.ndim == 4 and latent_features.shape[0] == 1:
        # Batch size is 1, keep as is
        pass
    else:
        # Other cases, keep as is
        pass

    # Now latent_features should have shape (n_samples, channels, height, width) or (n_samples, features)
    n_samples = latent_features.shape[1]

    # If latent_features has more than 2 dimensions, flatten the rest
    if latent_features.ndim > 2:
        # Flatten the features
        latent_features = latent_features.reshape(n_samples, -1)
        print("Shape after reshaping latent_features:", latent_features.shape)

    # Normalize features
    scaler = StandardScaler()
    latent_features_normalized = scaler.fit_transform(latent_features)
    print("Shape after normalization:", latent_features_normalized.shape)

    return latent_features_normalized


def process_pixel_wise_features(latent_features_list, labels_list, sample_size):
    """
    Processes pixel-wise features and labels for segmentation masks.

    Parameters:
    - latent_features_list: List of latent feature arrays.
    - labels_list: List of label arrays.
    - sample_size: Number of pixels to sample.

    Returns:
    - pixel_features_normalized: Normalized pixel-wise features.
    - pixel_labels_sampled: Corresponding pixel labels.
    """
    pixel_features = []
    pixel_labels = []

    for idx, (features_batch, labels_batch) in enumerate(
        zip(latent_features_list, labels_list)
    ):
        print(f"Processing batch {idx+1}/{len(latent_features_list)}")

        # Get batch size and dimensions
        channels, height, width = features_batch.shape

        # Reshape features and labels
        features_flat = features_batch.reshape(channels, -1)  # (channels, num_pixels)
        labels_flat = labels_batch.reshape(-1)  # (num_pixels)

        # Transpose features to (num_pixels, channels)
        features_flat = features_flat.transpose(1, 0)

        pixel_features.append(features_flat)  # (num_pixels, channels)
        pixel_labels.append(labels_flat)  # (num_pixels,)

    # Concatenate all data
    pixel_features = np.concatenate(pixel_features, axis=0)  # (total_pixels, channels)
    pixel_labels = np.concatenate(pixel_labels, axis=0)  # (total_pixels,)
    print("Total number of pixels:", pixel_features.shape[0])

    # Subsample pixels
    num_pixels = pixel_features.shape[0]
    sample_size = min(num_pixels, sample_size)
    indices = np.random.choice(num_pixels, size=sample_size, replace=False)
    pixel_features_sampled = pixel_features[indices]
    pixel_labels_sampled = pixel_labels[indices]
    print(f"Number of pixels after subsampling: {sample_size}")

    # Normalize features
    scaler = StandardScaler()
    pixel_features_normalized = scaler.fit_transform(pixel_features_sampled)
    print("Pixel features normalized.")

    return pixel_features_normalized, pixel_labels_sampled


'''
def process_pixel_wise_features(latent_features_list, labels_list, sample_size):
    """
    Processes pixel-wise features and labels for segmentation masks.

    Parameters:
    - latent_features_list: List of latent feature arrays.
    - labels_list: List of label arrays.
    - sample_size: Number of pixels to sample.

    Returns:
    - pixel_features_normalized: Normalized pixel-wise features.
    - pixel_labels_sampled: Corresponding pixel labels.
    """
    pixel_features = []
    pixel_labels = []

    for idx, (features_batch, labels_batch) in enumerate(
        zip(latent_features_list, labels_list)
    ):
        print(f"Processing batch {idx+1}/{len(latent_features_list)}")

        # Handle dual-real-valued tensors by merging the last dimension into channels
        if features_batch.ndim == 5 and features_batch.shape[-1] == 2:
            # Merge the last dimension (real and imaginary) into the channel dimension
            features_batch = features_batch.reshape(
                features_batch.shape[0],
                features_batch.shape[1] * features_batch.shape[4],
                features_batch.shape[2],
                features_batch.shape[3],
            )
            print(f"Reshaped dual-real-valued features_batch: {features_batch.shape}")
        elif features_batch.ndim == 4 and features_batch.shape[0] == 1:
            # Already merged real and imaginary parts
            pass
        else:
            raise ValueError(
                f"Unexpected shape for features_batch: {features_batch.shape}"
            )

        # Handle labels_batch
        if labels_batch.ndim == 2:
            # If labels have shape (height, width), add batch dimension
            labels_batch = np.expand_dims(labels_batch, axis=0)
            print(f"Added batch dimension to labels_batch: {labels_batch.shape}")
        elif labels_batch.ndim == 3 and labels_batch.shape[0] == 1:
            # Batch size is 1, keep as is
            pass
        else:
            raise ValueError(f"Unexpected shape for labels_batch: {labels_batch.shape}")

        # Now, features_batch should have shape (batch_size, channels, height, width)
        #       labels_batch should have shape (batch_size, height, width)

        # Check shapes
        assert (
            features_batch.ndim == 4
        ), f"Features should have shape (batch_size, channels, height, width), got {features_batch.shape}"
        assert (
            labels_batch.ndim == 3
        ), f"Labels should have shape (batch_size, height, width), got {labels_batch.shape}"

        # Get batch size and dimensions
        batch_size, channels, height, width = features_batch.shape

        # Reshape features and labels
        features_flat = features_batch.reshape(
            batch_size, channels, -1
        )  # (batch_size, channels, num_pixels)
        labels_flat = labels_batch.reshape(batch_size, -1)  # (batch_size, num_pixels)

        # Transpose features to (batch_size, num_pixels, channels)
        features_flat = features_flat.transpose(0, 2, 1)

        # Collect features and labels
        for i in range(batch_size):
            pixel_features.append(features_flat[i])  # (num_pixels, channels)
            pixel_labels.append(labels_flat[i])  # (num_pixels,)

    # Concatenate all data
    pixel_features = np.concatenate(pixel_features, axis=0)  # (total_pixels, channels)
    pixel_labels = np.concatenate(pixel_labels, axis=0)  # (total_pixels,)
    print("Total number of pixels:", pixel_features.shape[0])

    # Subsample pixels
    num_pixels = pixel_features.shape[0]
    sample_size = min(num_pixels, sample_size)
    indices = np.random.choice(num_pixels, size=sample_size, replace=False)
    pixel_features_sampled = pixel_features[indices]
    pixel_labels_sampled = pixel_labels[indices]
    print(f"Number of pixels after subsampling: {sample_size}")

    # Normalize features
    scaler = StandardScaler()
    pixel_features_normalized = scaler.fit_transform(pixel_features_sampled)
    print("Pixel features normalized.")

    return pixel_features_normalized, pixel_labels_sampled
'''


def visualize_latent_space(
    features, labels, n_components, path, wandb_log, class_colors
):
    """
    Applies dimensionality reduction and visualizes the latent space with discrete labels.

    Parameters:
    - features: NumPy array of processed latent features.
    - labels: NumPy array of labels corresponding to the features.
    - n_components: Number of dimensions for UMAP (2 or 3).
    - path: Path object where the HTML file will be saved.
    - wandb_log: Boolean indicating whether to log the visualization to Weights & Biases.
    - class_colors: Dictionary mapping label values to colors.

    Returns:
    - None (saves an interactive HTML plot of the latent space visualization).
    """

    # Dimensionality reduction
    reducer = umap.UMAP(n_components=n_components)
    embeddings = reducer.fit_transform(features)
    print("Shape of embeddings:", embeddings.shape)

    # Prepare DataFrame
    df = pd.DataFrame(
        embeddings, columns=[f"Component_{i+1}" for i in range(n_components)]
    )
    df["Label"] = labels.astype(int).astype(
        str
    )  # Ensure labels are strings for categorical mapping

    # Map labels to colors using class_colors
    df["Color"] = df["Label"].map(lambda x: class_colors.get(int(x), "grey"))

    # Check for missing color mappings
    missing_colors = df["Color"] == "grey"
    if missing_colors.any():
        missing_labels = df.loc[missing_colors, "Label"].unique()
        print(
            f"Warning: Missing color mappings for labels: {missing_labels}. These labels are colored grey."
        )

    # Create a scatter plot
    fig = go.Figure()

    if n_components == 2:
        # Plot data points
        fig.add_trace(
            go.Scattergl(
                x=df["Component_1"],
                y=df["Component_2"],
                mode="markers",
                marker=dict(
                    color=df["Color"],
                    size=5,
                    opacity=0.7,
                ),
                text=df["Label"],  # Hover text
                hovertemplate="Label: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>",
                showlegend=False,  # Hide the main trace in legend
            )
        )

        # Add legend entries manually
        for label, color in class_colors.items():
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                    ),
                    name=str(label),
                )
            )

        # Update layout
        fig.update_layout(
            title="UMAP Latent Space Visualization",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template="plotly_white",
        )

    elif n_components == 3:
        # Plot data points
        fig.add_trace(
            go.Scatter3d(
                x=df["Component_1"],
                y=df["Component_2"],
                z=df["Component_3"],
                mode="markers",
                marker=dict(
                    color=df["Color"],
                    size=3,
                    opacity=0.7,
                ),
                text=df["Label"],  # Hover text
                hovertemplate="Label: %{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
                showlegend=False,  # Hide the main trace in legend
            )
        )

        # Add legend entries manually
        for label, color in class_colors.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                    ),
                    name=str(label),
                )
            )

        # Update layout
        fig.update_layout(
            title="UMAP Latent Space Visualization",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            ),
            template="plotly_white",
        )
    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    # Define the output path
    path_html = path / "UMAP_latent_space.html"

    # Save the plot as an interactive HTML file
    fig.write_html(str(path_html))
    print(f"Visualization saved as {path_html}")

    # Log to Weights & Biases if required
    if wandb_log:
        wandb.log({f"umap": wandb.Html(str(path_html))})
        print("Visualization logged to Weights & Biases.")


def log_model_performance(hyperparameters, metrics, excel_file_path, task):
    """
    Logs model performance and hyperparameters into an Excel file for a specific task.
    Handles nested hyperparameters by creating hierarchical sub-columns.

    Parameters:
    - hyperparameters (dict): Dictionary containing the model's hyperparameters.
    - metrics (dict): Dictionary containing the model's performance metrics.
    - excel_file_path (str): Path to the Excel file where the data should be logged.
    - task (str): Name of the task (e.g., 'segmentation', 'classification') to be used as the sheet name.
    """
    # Extract run_id and remove "logging" from hyperparameters
    run_id = (
        hyperparameters.get("logging", {})
        .get("wandb", {})
        .get("run_id", "unknown_run_id")
    )
    excel_file_path = (
        os.path.dirname(os.path.dirname(hyperparameters["logging"]["logdir"]))
        + "/"
        + excel_file_path
    )
    del hyperparameters["logging"]
    del hyperparameters["data"]["characteristics"]

    log_entry = {"run_id": run_id}
    log_entry.update(hyperparameters)
    log_entry.update(metrics)

    # Extract structure of the nested dictionary
    tuples_structure, x_values = extract_structure(log_entry)

    # Create a MultiIndex for the columns from the tuples
    columns = pd.MultiIndex.from_tuples(tuples_structure)

    # Create a DataFrame with one row containing the extracted values
    log_df = pd.DataFrame([x_values], columns=columns)

    # Check if the Excel file exists
    if not os.path.exists(excel_file_path):
        # If the file does not exist, create a new file with the sheet for the task
        with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
            log_df.to_excel(writer, sheet_name=task, index=True)
        print(
            f"Created new Excel file and logged model performance for {task} task with run ID {run_id}"
        )
    else:
        # If the file exists, append the new row to the specified sheet
        with pd.ExcelWriter(
            excel_file_path, mode="a", if_sheet_exists="overlay", engine="openpyxl"
        ) as writer:
            if task in writer.sheets:
                # Append the data to the existing sheet
                start_row = writer.sheets[task].max_row
                log_df.to_excel(
                    writer,
                    sheet_name=task,
                    index=True,
                    header=False,
                    startrow=start_row,
                )
            else:
                # Create a new sheet and write the data with headers
                log_df.to_excel(writer, sheet_name=task, index=True)
        print(
            f"Logged model performance for {task} task with run ID {run_id} to {excel_file_path}"
        )


def extract_structure(nested_dict):
    from collections import defaultdict

    def find_max_depth(d, current_depth=0):
        """Helper function to find the maximum depth of the nested dictionary."""
        if isinstance(d, dict):
            return max(
                [find_max_depth(v, current_depth + 1) for v in d.values()],
                default=current_depth,
            )
        return current_depth

    def traverse_dict(d, current_path=(), parent=""):
        """Helper function to recursively traverse the dictionary and extract keys and values, avoiding key name conflicts."""
        for key, value in d.items():
            new_key = key

            # Detect if key needs disambiguation (same key appearing in different branches)
            if key in key_occurrences and key_occurrences[key] > 1:
                new_key = f"{parent}_{key}" if parent else key

            new_path = current_path + (new_key,)

            if isinstance(value, dict):
                yield from traverse_dict(value, new_path, parent=new_key)
            else:
                yield new_path, value

    # Step 1: Find the maximum depth
    max_depth = find_max_depth(nested_dict)

    # Step 2: Count key occurrences across all branches to detect duplicates
    key_occurrences = defaultdict(int)

    def count_keys(d):
        """Helper function to count occurrences of keys in the nested dictionary."""
        for key, value in d.items():
            key_occurrences[key] += 1
            if isinstance(value, dict):
                count_keys(value)

    count_keys(nested_dict)

    # Step 3: Traverse the dictionary and gather keys and values
    key_list = []
    value_list = []

    for path, value in traverse_dict(nested_dict):
        # Extend the path with empty strings to match the maximum depth
        padded_path = path + ("",) * (max_depth - len(path))
        key_list.append(padded_path)
        value_list.append(value)

    return key_list, value_list
