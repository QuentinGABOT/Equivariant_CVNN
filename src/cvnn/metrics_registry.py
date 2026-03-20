"""
Metrics registry for evaluation functions based on task and pipeline type.
Includes functional implementations of complex metrics (SSIM) to avoid circular dependencies.
"""
# Standard library imports
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Third-party imports
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from scipy import linalg
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

# Local import
from cvnn.utils import setup_logging
from cvnn.physics import (
    pauli_transform,
    h_alpha,
    cameron,
)
from cvnn.models.feature_extractors import MnistClassifier, MstarClassifier, MRIReconstructor, Cifar10Classifier
from cvnn.data_processing import dual_real_to_complex_transform

logger = setup_logging(__name__)

# ==============================================================================
# 1. Functional Metric Implementations (SSIM & Utils)
# ==============================================================================

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D gaussian kernel."""
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    """Creates a 2D gaussian window for SSIM."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim_calculation(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window: Variable, 
    window_size: int, 
    channel: int, 
    size_average: bool = True
) -> torch.Tensor:
    """Internal SSIM calculation using convolution."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window_size: int = 11, 
    size_average: bool = True
) -> torch.Tensor:
    """
    Computes Structural Similarity Index (SSIM) for real-valued images.
    Expects input shape (B, C, H, W).
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_calculation(img1, img2, window, window_size, channel, size_average)


def complex_ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window_size: int = 11, 
    size_average: bool = True
) -> torch.Tensor:
    """
    Computes SSIM on the magnitude of complex tensors.
    Expects complex input shape (B, C, H, W).
    """
    return ssim(torch.abs(img1), torch.abs(img2), window_size, size_average)


# ==============================================================================
# 2. Standard Metric Wrappers
# ==============================================================================

def mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error for real-valued data."""
    return F.mse_loss(predictions, targets).item()


def complex_mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error for complex-valued data."""
    diff = predictions - targets
    return torch.mean(torch.abs(diff) ** 2).item()


def psnr_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio for real-valued data."""
    mse = F.mse_loss(predictions, targets)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def complex_psnr_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio for complex-valued data."""
    mse = complex_mse_metric(predictions, targets) # returns float
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def ssim_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """SSIM metric wrapper."""
    return ssim(predictions, targets).item()


def complex_ssim_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Complex SSIM metric wrapper."""
    return complex_ssim(predictions, targets).item()


def accuracy_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy. Expects logits/softmax as predictions."""
    # If predictions are already indices (B,), skip argmax
    if predictions.dim() == targets.dim() + 1: 
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
        
    correct = (pred_classes == targets).float()
    return torch.mean(correct).item()


def iou_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Intersection over Union for segmentation."""
    if predictions.dim() == targets.dim() + 1:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
        
    intersection = torch.logical_and(pred_classes, targets)
    union = torch.logical_or(pred_classes, targets)
    # Handle division by zero if union is empty
    if torch.sum(union) == 0:
        return 1.0 if torch.sum(intersection) == 0 else 0.0
    
    iou = torch.sum(intersection) / torch.sum(union)
    return iou.item()


def dice_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Dice coefficient for segmentation."""
    if predictions.dim() == targets.dim() + 1:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
        
    intersection = torch.logical_and(pred_classes, targets).float().sum()
    union = pred_classes.float().sum() + targets.float().sum()
    if union == 0:
        return 1.0
    dice = (2 * intersection) / union
    return dice.item()

def phase_weighted_error_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Phase-Weighted Error (PWE): Penalizes phase errors proportional to signal magnitude.
    Formula: Mean( |target| * |angle(target) - angle(pred)|^2 )
    """
    # Ensure complex
    if not torch.is_complex(predictions) or not torch.is_complex(targets):
        return 0.0 # Or raise error
    
    mag = torch.abs(targets)
    # Difference angulaire wrapée entre -pi et pi
    diff_angle = torch.angle(targets) - torch.angle(predictions)
    diff_angle = torch.atan2(torch.sin(diff_angle), torch.cos(diff_angle))
    
    # Weighted error
    pwe = torch.mean(mag * (diff_angle ** 2))
    return pwe.item()

def complex_correlation_coefficient_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Complex Correlation Coefficient (CCC): Measures linear dependence and phase sync.
    Computes the cosine similarity between the two complex vectors formed by flattening the inputs.
    """
    # Flatten batch and spatial dims to treat them as single global vectors
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # CCC = |<y, x>| / (||y|| * ||x||)
    # Note: peu importe quel vecteur est conjugué car on prend le module |.| à la fin
    numerator = torch.abs(torch.sum(target_flat.conj() * pred_flat))
    
    norm_pred = torch.sqrt(torch.sum(torch.abs(pred_flat)**2))
    norm_target = torch.sqrt(torch.sum(torch.abs(target_flat)**2))
    
    denominator = norm_pred * norm_target
    
    if denominator == 0:
        return 0.0
        
    return (numerator / denominator).item()

def _compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[Union[List[int], Dict[int, str]]] = None,
) -> Dict[str, Any]:
    """
    Generic helper to compute a full classification report.
    Automatically infers classes if class_labels is None.
    """
    # 1. Infer labels if not provided
    if class_labels is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        class_labels = sorted(unique_labels.tolist())
        class_labels_values = class_labels
    elif isinstance(class_labels, dict):
        class_labels_values = list(class_labels.keys())
    else:
        class_labels_values = class_labels

    # 2. Global metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = np.nan

    # 3. Averaged metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels_values, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels_values, average="weighted", zero_division=0
    )

    # 4. Per-class metrics
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels_values, average=None, zero_division=0
    )

    # 5. Confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred, labels=class_labels_values)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_labels_values, normalize="true")

    # 6. Distributions
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    dist_true = {int(k): int(v) for k, v in zip(unique_true, counts_true)}
    dist_pred = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}

    def get_name(idx):
        if isinstance(class_labels, dict):
            return class_labels.get(idx, str(idx))
        return str(idx)

    return {
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "matthews_corrcoef": float(mcc) if not np.isnan(mcc) else None,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weight),
        "recall_weighted": float(r_weight),
        "f1_weighted": float(f1_weight),
        "per_class_metrics": {
            int(label): {
                "name": get_name(label),
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(class_labels_values, precision, recall, fscore, support)
        },
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "class_distribution_original": dist_true,
        "class_distribution_generated": dist_pred,
        "class_labels": class_labels_values,
    }


def classification_report_wrapper(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    """Adaptateur pour rendre compute_classification_report compatible avec le registre."""
    # 1. Conversion Logits -> Indices (si nécessaire)
    if predictions.ndim > targets.ndim: # Ex: Preds (B, C, H, W) vs Targets (B, H, W)
        predictions = torch.argmax(predictions, dim=1)
    
    # 2. Conversion Tensor GPU -> Numpy CPU
    y_pred = predictions.detach().cpu().numpy().flatten()
    y_true = targets.detach().cpu().numpy().flatten()
    
    # 3. Appel de la fonction originale
    return _compute_classification_report(y_true, y_pred)

# ==============================================================================
# 3. Registry Infrastructure
# ==============================================================================

# Metrics registry structure:
# {task: {pipeline_type: {metric_name: metric_function}}}
_METRICS_REGISTRY: Dict[str, Dict[str, Dict[str, Callable]]] = {}


def register_metric(
    task: str, pipeline_type: str, metric_name: str, metric_function: Callable
) -> None:
    """Register a metric function for a specific task and pipeline type."""
    if task not in _METRICS_REGISTRY:
        _METRICS_REGISTRY[task] = {}

    if pipeline_type not in _METRICS_REGISTRY[task]:
        _METRICS_REGISTRY[task][pipeline_type] = {}

    _METRICS_REGISTRY[task][pipeline_type][metric_name] = metric_function
    logger.debug(f"Registered metric: {task}.{pipeline_type}.{metric_name}")


def get_metric_function(
    task: str, pipeline_type: str, metric_name: str
) -> Optional[Callable]:
    """Get a metric function for specific task and pipeline type."""
    try:
        return _METRICS_REGISTRY[task][pipeline_type][metric_name]
    except KeyError:
        # Fallback: check if metric exists in 'real_real' if current pipeline missing
        if pipeline_type != "real_real":
             try:
                 return _METRICS_REGISTRY[task]["real_real"][metric_name]
             except KeyError:
                 pass
        return None


def get_available_metrics(task: str, pipeline_type: str) -> List[str]:
    """Get list of available metrics for a task and pipeline type."""
    try:
        return list(_METRICS_REGISTRY[task][pipeline_type].keys())
    except KeyError:
        return []


def _register_standard_metrics():
    """Register commonly used metrics for different tasks and pipeline types."""
    
    # 1. Reconstruction & Generation
    # Real pipelines
    real_pipes = ["real_real", "complex_amplitude_real"]
    real_metrics = {
        "mse": mse_metric, 
        "psnr": psnr_metric, 
        "ssim": ssim_metric, 
        "mae": lambda p, t: F.l1_loss(p, t).item(),
    }
    
    # Complex pipelines
    complex_pipes = ["complex_dual_real", "complex", "split", "widely_linear"]
    complex_metrics = {
        "mse": complex_mse_metric,
        "psnr": complex_psnr_metric,
        "ssim": complex_ssim_metric,
        "mae": lambda p, t: torch.mean(torch.abs(p - t)).item(),
        "pwe": phase_weighted_error_metric,
        "ccc": complex_correlation_coefficient_metric,
    }

    for task in ["reconstruction", "generation"]:
        for pipe in real_pipes:
            for name, func in real_metrics.items():
                register_metric(task, pipe, name, func)
        
        for pipe in complex_pipes:
            for name, func in complex_metrics.items():
                register_metric(task, pipe, name, func)

    # 2. Segmentation & Classification
    # These metrics generally operate on argmaxed indices, so they are shared
    cls_metrics = {
        "accuracy": accuracy_metric,
        "iou": iou_metric,
        "dice": dice_metric,
        "report": classification_report_wrapper,
    }
    
    all_pipelines = real_pipes + complex_pipes
    for task in ["segmentation", "classification"]:
        for pipe in all_pipelines:
            for name, func in cls_metrics.items():
                register_metric(task, pipe, name, func)

# Run registration on import
_register_standard_metrics()


# ==============================================================================
# 4. OOP Interface (The Evaluator's Entry Point)
# ==============================================================================

class MetricsRegistry:
    """
    Unified metrics registry interface for the Evaluator.
    Handles pipeline inference and batch metric computation.
    """

    def __init__(self, task: str, cfg: Any):
        """
        Args:
            task: Task name ("reconstruction", "segmentation", etc.)
            cfg: Configuration object (OmegaConf or dict)
        """
        self.task = task
        self.cfg = cfg
        self.pipeline_type = self._infer_pipeline_type(cfg)
        
    def _infer_pipeline_type(self, cfg: Any) -> str:
        """Determines pipeline type from config."""
        layer_mode = cfg["model"].get("layer_mode")
        if layer_mode == "real":
            real_pipeline_type = cfg["data"].get("real_pipeline_type")
            return real_pipeline_type
        else:
            return layer_mode

    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        metric_subset: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for predictions vs targets.
        
        Args:
            predictions: Tensor (B, C, H, W)
            targets: Tensor (B, C, H, W)
            metric_subset: Optional list of metric names to run (overrides config).
        """
        results = {}

        # 1. Determine which metrics to run
        if metric_subset is not None:
            # Force usage of specific metrics provided by caller
            metric_names = metric_subset
        else:
            # Default behavior: use config or all available
            eval_cfg = self.cfg.get("evaluation", {}) if isinstance(self.cfg, dict) else self.cfg.evaluation
            metric_names = getattr(eval_cfg, "metrics", None)
            
            if not metric_names:
                metric_names = get_available_metrics(self.task, self.pipeline_type)

        # 2. Compute loop
        for name in metric_names:
            func = get_metric_function(self.task, self.pipeline_type, name)
            if func:
                try:
                    val = func(predictions, targets)
                    if isinstance(val, dict):
                        results.update(val)
                    else:
                        results[name] = float(val)
                except Exception as e:
                    # Silence specific errors if dimensions mismatch for certain metrics
                    logger.warning(f"Failed to compute metric '{name}': {e}")
            else:
                pass 
                # logger.debug(f"Metric '{name}' not found.")

        return results

# ==============================================================================
# 5. Full image error reconstruction analysis
# ==============================================================================

def angular_distance(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute the angular distance between two complex-valued images.

    The angular distance measures the phase difference between corresponding pixels
    in two complex images, with results normalized to the range [-π, π].

    Args:
        image1: First complex-valued image (numpy array)
        image2: Second complex-valued image (numpy array)

    Returns:
        Angular distance array with values in [-π, π]

    Raises:
        ValueError: If input arrays have different shapes or are not complex-valued
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Input shapes must match: {image1.shape} vs {image2.shape}")

    if not (np.iscomplexobj(image1) and np.iscomplexobj(image2)):
        raise ValueError("Both input images must be complex-valued")

    # Compute phase difference and normalize to [-π, π]
    diff = np.angle(image1) - np.angle(image2) + np.pi
    angular_dist = np.mod(diff, 2 * np.pi) - np.pi
    return angular_dist

def compute_reconstruction_errors(
    original: np.ndarray,
    reconstructed: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Compute pixel-wise reconstruction errors between original and reconstructed images.
    
    Args:
        original: Original image array (H, W) or (H, W, C)
        reconstructed: Reconstructed image array (H, W) or (H, W, C)
        cfg: Configuration dictionary
    Returns:
        Dictionary with error maps:
            
    """
    if cfg["data"].get("type").lower() == "polsar":
        amp_diff = (np.abs(original) - np.abs(reconstructed)).flatten()
        ang_dist = angular_distance(original, reconstructed).flatten()
    else:
        amp_diff = (original - reconstructed).flatten()
        ang_dist = np.array([])  # No phase for non-complex data
    return {
        "amplitude_difference": amp_diff,
        "angular_distance": ang_dist,
    }
    
# ==============================================================================
# 6. PolSAR-decomposition Classification Metrics
# ==============================================================================

def compute_h_alpha_metrics(
    image1: np.ndarray,
    image2: np.ndarray,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for H-alpha classification comparison.
    """
    if class_labels is None:
        # Standard H-alpha classes (1 to 9, except 3 which is physically impossible in some zones)
        class_labels = {
        1: "Complex structures",
        2: "Random anisotropic scatterers",
        4: "Double reflection propagation effects",
        5: "Anisotropic particles",
        6: "Random surfaces",
        7: "Dihedral reflector",
        8: "Dipole",
        9: "Bragg surface",
    }
    image1 = pauli_transform(image1)
    image2 = pauli_transform(image2)

    h_alpha1 = h_alpha(image1)
    h_alpha2 = h_alpha(image2)
    return _compute_classification_report(
        h_alpha1.flatten(),
        h_alpha2.flatten(),
        class_labels,
    )

def compute_cameron_metrics(
    image1: np.ndarray,
    image2: np.ndarray,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for Cameron classification comparison.
    """
    if class_labels is None:
        class_labels = {
        1: "Non-reciprocal",
        2: "Asymmetric",
        3: "Left helix",
        4: "Right helix",
        5: "Symmetric",
        6: "Trihedral",
        7: "Dihedral",
        8: "Dipole",
        9: "Cylinder",
        10: "Narrow dihedral",
        11: "Quarter-wave",
    }
    
    cameron1 = cameron(image1)
    cameron2 = cameron(image2)

    return _compute_classification_report(
        cameron1.flatten(),
        cameron2.flatten(),
        class_labels,
    )

# ==============================================================================
# 7. Custom FID
# ==============================================================================

class CustomFID:
    """
    Calculate the FID distance using a domain-specific feature extractor, instead of InceptionV3.
    """
    def __init__(self, dataset_name: str, device: torch.device, checkpoint_dir: str = "checkpoints/classifiers"):
        self.device = device
        self.dataset_name = dataset_name
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 1. Load the correct Extractor (Feature Extractor)
        self.model, self.input_size = self._load_extractor()
        self.model.to(device)
        self.model.eval()

    def _load_extractor(self):
        """Select and load the trained feature extractor."""
        input_size = None # Expected input size for the extractor (H, W)
        
        if "mnist" in self.dataset_name:
            model = MnistClassifier(num_classes=10)
            fname = "mnist_classifier.pt"
            input_size = (32, 32)
        elif "cifar10" in self.dataset_name:
            model = Cifar10Classifier(num_classes=10)
            fname = "cifar10_classifier.pt"
            input_size = (32, 32)
        elif "sar" in self.dataset_name or "mstar" in self.dataset_name:
            model = MstarClassifier(num_classes=10) 
            fname = "mstar_classifier.pt"
            input_size = (32, 32)
        elif "flatcmrxrecon" in self.dataset_name:
            model = MRIReconstructor() 
            fname = "mri_dae.pt"
            input_size = (32, 32)
        else:
            raise ValueError(f"No extractor available for dataset {self.dataset_name}")

        path = self.checkpoint_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}.")
            
        # Load weights
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        logger.info(f"Custom FID Extractor loaded: {fname}")
            
        return model, input_size

    def get_activations(self, data_loader: DataLoader, num_samples: int = None):
        pred_arr = []
        total_count = 0
                
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                if not x.is_complex():
                    x = dual_real_to_complex_transform(x)

                # Extraction
                features = self.model.get_features(x)
                
                pred_arr.append(features.cpu().numpy())
                total_count += x.size(0)
                
                if num_samples is not None and total_count >= num_samples:
                    break

        activations = np.concatenate(pred_arr, axis=0)
        if num_samples is not None:
            activations = activations[:num_samples]
        logger.info(f"Extracted {activations.shape} activations for FID computation.")
        return activations

    def compute_fid(self, real_loader, fake_loader, num_samples=2000):       
        act1 = self.get_activations(real_loader, num_samples)
        act2 = self.get_activations(fake_loader, num_samples)
        print("Here")
        input()
        mu1, sigma1 = self._compute_augmented_statistics(act1)
        print("Real statistics computed: mu1 shape:", mu1.shape, "sigma1 shape:", sigma1.shape)
        input()
        mu2, sigma2 = self._compute_augmented_statistics(act2)
        print("Fake statistics computed: mu2 shape:", mu2.shape, "sigma2 shape:", sigma2.shape)
        input()
        fid_value = self._calculate_complex_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

    def _compute_augmented_statistics(self, features):
        """
        C = E[(z-mu)(z-mu)^H] (Covariance)
        P = E[(z-mu)(z-mu)^T] (Pseudo-covariance)
        Sigma_aug = [[C, P], [P*, C*]]
        """
            
        N, D = features.shape
        mu = np.mean(features, axis=0)
        
        features_centered = features - mu
        
        with threadpool_limits(limits=1, user_api='blas'):
            C = np.cov(features, rowvar=False) 
            P = (features_centered.T @ features_centered) / (N - 1)
            
            C = np.atleast_2d(C)
            P = np.atleast_2d(P)
            
            top_left = C
            top_right = P
            bottom_left = P.conj()
            bottom_right = C.conj()
            
            sigma_aug = np.block([[top_left, top_right], [bottom_left, bottom_right]])
            mu_aug = np.concatenate([mu, mu.conj()])
        
        return mu_aug, sigma_aug

    def _calculate_complex_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        mean_term = diff.conj().dot(diff)

        with threadpool_limits(limits=1, user_api='blas'):        
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)                          
            tr_covmean = np.trace(covmean)  
            fid_value = (mean_term + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
         
        return float(np.real(fid_value))