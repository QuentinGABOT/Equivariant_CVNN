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
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

### Local imports
from cvnn.utils import setup_logging, count_model_parameters
from cvnn.data_processing import dual_real_to_complex_transform
from cvnn.metrics_registry import MetricsRegistry, CustomFID

logger = setup_logging(__name__)

# ==============================================================================
# Helper Class for CustomFID
# ==============================================================================

class _FakeDataLoader:
    """
    Wrapper interne pour simuler un DataLoader infini qui génère des images depuis le VAE.
    Nécessaire pour l'interface de CustomFID.
    """
    def __init__(self, model, batch_size, device):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            samples = self.model.sample(self.batch_size)
            return samples.to(self.device)
        
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
    
class GenerationEvaluator(ReconstructionEvaluator):
    def __init__(self, model, test_loader, cfg, task, device=None):
        super().__init__(model, test_loader, cfg, task, device)
        
        # 1. Standard Inception FID (Baseline)
        self.fid = FrechetInceptionDistance(feature=64, normalize=True).to(self.device)
        
        # 2. Custom Domain-Aware FID (Ours)
        if CustomFID is not None:
            dataset_name = cfg.get("data").get("dataset").get("name").lower()          
            try:
                self.custom_fid = CustomFID(dataset_name=dataset_name, device=self.device)
                logger.info(f"CustomFID initialized for dataset: {dataset_name}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"CustomFID skipped: {e} (Ensure you ran train_feature_extractor.py)")
            except Exception as e:
                logger.warning(f"CustomFID failed to init: {e}")

    def evaluate(self) -> Dict[str, float]:
        # 1. Run the Hybrid Reconstruction Eval (Spectral + Spatial)
        metrics = super().evaluate()
        
        # 2. Add Generative Metrics (FID Standard + Custom)
        gen_metrics = self.evaluate_generative_metrics(num_samples=5000)
        
        metrics["metrics"].update(gen_metrics)
                
        return metrics

    def _prepare_for_inception(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts CVNN output to format expected by InceptionV3.
        """
        if not torch.is_complex(tensor):
            tensor = dual_real_to_complex_transform(tensor)

        if torch.is_complex(tensor):
            if self._is_spectral_data():
                 tensor = torch.fft.ifft2(torch.fft.ifftshift(tensor)).abs()
            else:
                tensor = tensor.abs()

        B, C, H, W = tensor.shape
        flat = tensor.view(B, -1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)

        if C == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif C > 3:
            tensor = tensor[:, :3, :, :] 
        
        elif C == 2:
             tensor = tensor.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        return tensor

    def evaluate_generative_metrics(self, num_samples=5000) -> Dict[str, float]:
        results = {}
        
        self.fid.reset()
        
        # Collect Stats
        with torch.no_grad():
            # Real
            for i, batch in enumerate(self.loader):
                if i >= num_samples: break
                real_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                real_imgs = real_imgs.to(self.device)
                real_rgb = self._prepare_for_inception(real_imgs)
                self.fid.update(real_rgb, real=True)

            # Fake
            required_fake = self.fid.real_features_num_samples
            total_fake = 0
            while total_fake < required_fake:
                batch_size = min(self.loader.batch_size, required_fake - total_fake)
                fake_raw = self.model.sample(batch_size).to(self.device)
                fake_rgb = self._prepare_for_inception(fake_raw)
                self.fid.update(fake_rgb, real=False)
                total_fake += batch_size

            results["fid_imagenet_baseline"] = self.fid.compute().item()

        # --- B. Custom Domain-Aware FID ---
        if self.custom_fid is not None:
            fake_loader_wrapper = _FakeDataLoader(self.model, self.loader.batch_size, self.device)
            
            custom_score = self.custom_fid.compute_fid(
                real_loader=self.loader,
                fake_loader=fake_loader_wrapper,
                num_samples=5000
            )
            results["fid_custom"] = custom_score

        return results
    
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
    elif task == "generation":
        return GenerationEvaluator(model, loader, cfg, task, device)
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

def compute_linear_probing(model, train_loader, test_loader, device, max_samples=5000) -> Optional[float]:
    """
    Trains a Logistic Regression on train_loader latents and evaluates on test_loader.
    Returns Accuracy (0.0 to 1.0).
    """   
    # 1. Extract Train Latents
    X_train, y_train = extract_latents_for_probing(model, train_loader, device, max_samples=max_samples)
    if X_train is None or y_train is None:
        logger.warning("Linear Probing skipped: No labels found in training data (Unsupervised dataset?).")
        return None

    # 2. Extract Test Latents
    X_test, y_test = extract_latents_for_probing(model, test_loader, device, max_samples=max_samples)
    if X_test is None or y_test is None:
        logger.warning("Linear Probing skipped: No labels found in test data.")
        return None

    # 3. Train Classifier
    # Max iter increased for convergence
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Linear Probing Failed during fit: {e}")
        return None

    # 4. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return acc