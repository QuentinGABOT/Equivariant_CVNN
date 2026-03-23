"""
Unified evaluation module for CVNN.
Handles Reconstruction, Generation, Segmentation, and Classification metrics.
For full image reconstruction/inference, see cvnn.inference.
"""

### Standard library imports
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

### Third-party imports
import torch
import numpy as np

### Local imports
from cvnn.utils import setup_logging, count_model_parameters
from cvnn.data_processing import dual_real_to_complex_transform
from cvnn.metrics_registry import MetricsRegistry

logger = setup_logging(__name__)

        
# ==============================================================================
# 1. Base Evaluator
# ==============================================================================

class BaseEvaluator(ABC):
    """Abstract base class for all evaluation tasks."""
    def __init__(
        self, 
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader, 
        cfg: Dict[str, Any], 
        task: str, 
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.loader = test_loader
        self.cfg = cfg
        self.task = task
        
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        self.registry = MetricsRegistry(task, cfg)
        self._log_model_stats()

    def _log_model_stats(self):
        return count_model_parameters(self.model)

    @abstractmethod
    def process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (predictions, targets) for metric computation."""
        pass

    def compute_consistency(self, inputs: torch.Tensor) -> float:
        return 0.0

    def evaluate(self) -> Dict[str, float]:
        """Main evaluation loop."""
        logger.info(f"Starting {self.task} evaluation...")
        metrics = {}
        all_preds = []
        all_targets = []
        consistency_scores = []
        check_invariance = self.cfg.get("evaluation", {}).get("check_invariance", False)
        
        with torch.no_grad():
            for batch in self.loader:
                preds, targets = self.process_batch(batch)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                
                if check_invariance:
                    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    consistency_scores.append(self.compute_consistency(inputs.to(self.device)))

        if not all_preds:
            return {}

        full_preds = torch.cat(all_preds, dim=0)
        full_targets = torch.cat(all_targets, dim=0)
        
        metrics["metrics"] = self.registry.compute_metrics(full_preds, full_targets)
        
        if consistency_scores:
            metrics["consistency"] = {"circular_shift": float(np.mean(consistency_scores))}
        
        metrics["stats"] = self._log_model_stats()
                        
        return metrics


# ==============================================================================
# 2. Task-Specific Evaluators
# ==============================================================================

class ReconstructionEvaluator(BaseEvaluator):
    def process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts RAW outputs. 
        We return the raw complex data (Spectral for FFT-MNIST, Spatial for SAR/MRI).
        """
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = inputs.to(self.device)
        
        outputs = self.model(inputs)
        if isinstance(outputs, tuple): outputs = outputs[0]
            
        targets = inputs
        
        # Standardize type structure (e.g. Dual-Channel Real -> Complex)
        pipeline_type = self.registry.pipeline_type
        if pipeline_type == "complex_dual_real":
            outputs = dual_real_to_complex_transform(outputs)
            targets = dual_real_to_complex_transform(targets)
            
        return outputs, targets

    def compute_consistency(self, inputs: torch.Tensor) -> float:
        """
        Computes consistency by measuring MSE between original and shifted reconstructions.
        No labels needed - suitable for reconstruction tasks.
        """
        shift = (np.random.randint(1, 8), np.random.randint(1, 8))
        
        # Original reconstruction
        out_orig = self.model(inputs)
        if isinstance(out_orig, tuple): out_orig = out_orig[0]
        
        # Shifted input and its reconstruction
        inputs_shifted = torch.roll(inputs, shifts=shift, dims=(-2, -1))
        out_shifted = self.model(inputs_shifted)
        if isinstance(out_shifted, tuple): out_shifted = out_shifted[0]
        
        # Shift prediction back to original space
        out_shifted_back = torch.roll(out_shifted, shifts=(-shift[0], -shift[1]), dims=(-2, -1))
        
        # Compute MSE between original and shifted-back prediction
        mse = torch.norm(out_orig.cpu() - out_shifted_back.cpu()).item()
        
        return mse

    def _is_spectral_data(self) -> bool:
        """
        Detects if the raw data is in the Frequency domain (requiring iFFT for visualization).
        """
        # 1. Check explicit evaluation config
        eval_domain = self.cfg.get("evaluation", {}).get("domain", None)
        if eval_domain == "spectral": return True
        if eval_domain == "spatial": return False
        
        # 2. Infer from dataset type
        d_type = self.cfg.get("data", {}).get("type", "").lower()
        if "fft" in d_type or "spectral" in d_type:
            return True
            
        # Default to Spatial (SAR, MRI, standard images)
        return False

    def evaluate(self) -> Dict[str, float]:
        """
        Universal Hybrid Evaluation:
        - Domain 1 (Complex/Physics): Validates phase & amplitude on raw data.
        - Domain 2 (Spatial/Perceptual): Validates visual quality (SSIM/FID) on magnitude.
        """
        logger.info(f"Starting {self.task} evaluation (Hybrid Physics+Perceptual)...")
        
        # 1. Collect RAW Data
        all_preds = []
        all_targets = []
        consistency_scores = []
        check_invariance = self.cfg.get("evaluation", {}).get("check_invariance", False)
        
        with torch.no_grad():
            for batch in self.loader:
                preds, targets = self.process_batch(batch)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                
                if check_invariance:
                    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    consistency_scores.append(self.compute_consistency(inputs.to(self.device)))

        if not all_preds:
            return {}

        # Concatène sur CPU (Raw Data)
        full_preds_raw = torch.cat(all_preds, dim=0)
        full_targets_raw = torch.cat(all_targets, dim=0)
        
        metrics = {"metrics": {}}

        # =========================================================
        # DOMAIN 1: COMPLEX PHYSICS (Raw Output)
        # =========================================================
        # SAR/MRI: Measures phase fidelity of pixels.
        # FFT-MNIST: Measures phase fidelity of frequencies.
        # Metrics: PWE, CCC, Complex-MSE
        
        phys_metrics = ["mse", "pwe", "ccc"]
        
        raw_results = self.registry.compute_metrics(
            full_preds_raw, 
            full_targets_raw, 
            metric_subset=phys_metrics
        )
        
        # Prefix "complex_" or "spectral_" depending on context logic if needed,
        # but "raw_" or "physics_" is clearer. Let's keep your convention or use "complex_".
        for k, v in raw_results.items():
            metrics["metrics"][f"complex_{k}"] = v

        # =========================================================
        # DOMAIN 2: SPATIAL MAGNITUDE (Perceptual)
        # =========================================================
        # Used for SSIM, PSNR, FID.
                
        if self._is_spectral_data():
            # CASE A: Spectral Data (FFT-MNIST) -> Needs iFFT
            # Note: ifftshift handles centering
            full_preds_spatial = torch.fft.ifft2(torch.fft.ifftshift(full_preds_raw)).abs()
            full_targets_spatial = torch.fft.ifft2(torch.fft.ifftshift(full_targets_raw)).abs()
        else:
            # CASE B: Spatial Data (SAR, MRI) -> Just Magnitude
            # Raw data is already spatial complex pixels.
            full_preds_spatial = full_preds_raw.abs()
            full_targets_spatial = full_targets_raw.abs()
        
        spat_metrics = ["ssim", "psnr", "mse"] # MSE Spatial
        
        spat_results = self.registry.compute_metrics(
            full_preds_spatial, 
            full_targets_spatial,
            metric_subset=spat_metrics
        )
        
        for k, v in spat_results.items():
            metrics["metrics"][f"spatial_{k}"] = v

        # =========================================================
        # Finalize
        # =========================================================
        if consistency_scores:
            metrics["consistency"] = {"circular_shift": float(np.mean(consistency_scores))}
        
        metrics["stats"] = self._log_model_stats()
        
        return metrics
    
class SegmentationEvaluator(BaseEvaluator):
    def process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple): outputs = outputs[1] 
        preds = torch.argmax(outputs, dim=1)
        return preds, targets

    def compute_consistency(self, inputs: torch.Tensor) -> float:
        shift = (np.random.randint(1, 8), np.random.randint(1, 8))
        out_orig = self.model(inputs)
        if isinstance(out_orig, tuple): out_orig = out_orig[1]
        
        inputs_shifted = torch.roll(inputs, shifts=shift, dims=(-2, -1))
        out_shifted = self.model(inputs_shifted)
        if isinstance(out_shifted, tuple): out_shifted = out_shifted[1]
        
        pred_orig = torch.argmax(out_orig, dim=1).cpu()
        pred_shifted = torch.argmax(out_shifted, dim=1).cpu()
        pred_shifted_back = torch.roll(pred_shifted, shifts=(-shift[0], -shift[1]), dims=(-2, -1))
        return (pred_orig == pred_shifted_back).float().mean().item()


class ClassificationEvaluator(BaseEvaluator):
    def process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple): outputs = outputs[1]
        return outputs, targets

    def compute_consistency(self, inputs: torch.Tensor) -> float:
        shift = (np.random.randint(1, 8), np.random.randint(1, 8))
        out_orig = self.model(inputs)
        if isinstance(out_orig, tuple): out_orig = out_orig[1]
        
        inputs_shifted = torch.roll(inputs, shifts=shift, dims=(-2, -1))
        out_shifted = self.model(inputs_shifted)
        if isinstance(out_shifted, tuple): out_shifted = out_shifted[1]
        
        return (torch.argmax(out_orig,1).cpu() == torch.argmax(out_shifted,1).cpu()).float().mean().item()


def get_evaluator(task: str, model, loader, cfg, device=None) -> BaseEvaluator:
    if task == "reconstruction":
        return ReconstructionEvaluator(model, loader, cfg, task, device)
    elif task == "segmentation":
        return SegmentationEvaluator(model, loader, cfg, task, device)
    elif task == "classification":
        return ClassificationEvaluator(model, loader, cfg, task, device)
    else:
        raise ValueError(f"Unknown task: {task}")

def evaluate(task: str, model, test_loader, cfg, device=None, **kwargs) -> Dict[str, float]:
    evaluator = get_evaluator(task, model, test_loader, cfg, device)
    return evaluator.evaluate()


def extract_latents_for_probing(model, loader, device, max_samples=None):
    """
    Extracts latent vectors (mu) and labels from a dataloader.
    Handles Complex -> Real concatenation for Scikit-Learn.
    """
    latents_list = []
    labels_list = []
    model.eval()
    
    total_count = 0  # Compteur d'échantillons
    
    with torch.no_grad():
        for batch in loader:
            # Handle (x, y) or just x
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x = batch
                y = None
                
            x = x.to(device)
            
            # --- Extract Latent (mu) ---
            if hasattr(model, "encode"):
                res = model.encode(x)
                if isinstance(res, tuple): mu = res[0]
                else: mu = res 
            else:
                out = model(x)
                if isinstance(out, tuple) and len(out) >= 2:
                    mu = out[1]
                else:
                    return None, None

            # --- Prepare for Sklearn (Complex -> Real Concat) ---
            if torch.is_complex(mu):
                mu = mu.reshape(mu.size(0), -1) 
                mu_flat = torch.cat([mu.real, mu.imag], dim=1)
            else:
                mu = mu.reshape(mu.size(0), -1)
                mu_flat = mu
                
            latents_list.append(mu_flat.cpu().numpy())
            
            if y is not None:
                labels_list.append(y.cpu().numpy())
            
            total_count += x.size(0)
            if max_samples is not None and total_count >= max_samples:
                break

    if not latents_list:
        return None, None

    X = np.concatenate(latents_list, axis=0)
    y = np.concatenate(labels_list, axis=0) if labels_list else None
    
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[:max_samples]
        if y is not None:
            y = y[:max_samples]
    
    return X, y