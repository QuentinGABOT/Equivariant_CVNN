# src/cvnn/data_statistics.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
from cvnn.utils import setup_logging

logger = setup_logging(__name__)

def compute_dataset_statistics(
    dataset: Dataset, 
    batch_size: int = 16, 
    num_workers: int = 0,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute mean, std, min, max, and covariance matrix for a dataset.
    Iterates over the dataset (with transforms active) to get the statistics
    of the actual data fed to the model.
    """
    logger.info("Recomputing dataset statistics (this may take a while)...")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize accumulators
    n_channels = None
    min_vals = None
    max_vals = None
    
    sum_real = None
    sum_imag = None
    sumsq_real = None
    sumsq_imag = None
    sum_cross = None # Sum of real * imag
    total_pixels = 0
    
    samples_processed = 0

    for batch in tqdm(loader, desc="Computing stats"):
        # Handle tuple (data, target) or simple data
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
            
        if isinstance(data, dict):
            data = data.get("inputs", list(data.values())[0])

        # Ensure numpy (B, C, H, W)
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if data.ndim == 3: # (B, H, W) -> add channel dim
            data = data[:, None, :, :]
            
        B, C, H, W = data.shape
        
        # Initialize if first batch
        if n_channels is None:
            n_channels = C
            min_vals = np.full(C, np.inf)
            max_vals = np.full(C, -np.inf)
            sum_real = np.zeros(C, dtype=np.float64)
            sum_imag = np.zeros(C, dtype=np.float64)
            sumsq_real = np.zeros(C, dtype=np.float64)
            sumsq_imag = np.zeros(C, dtype=np.float64)
            sum_cross = np.zeros(C, dtype=np.float64)

        # Min/Max (using magnitude for complex)
        if np.iscomplexobj(data):
            mag = np.abs(data)
            # Flatten spatial dims for stats
            current_min = mag.reshape(B, C, -1).min(axis=2).min(axis=0)
            current_max = mag.reshape(B, C, -1).max(axis=2).max(axis=0)
            
            real_part = data.real
            imag_part = data.imag
        else:
            current_min = data.reshape(B, C, -1).min(axis=2).min(axis=0)
            current_max = data.reshape(B, C, -1).max(axis=2).max(axis=0)
            
            real_part = data
            imag_part = np.zeros_like(data) # Imaginary part is 0

        min_vals = np.minimum(min_vals, current_min)
        max_vals = np.maximum(max_vals, current_max)
        
        # Accumulate sums
        sum_real += real_part.sum(axis=(0, 2, 3))
        sum_imag += imag_part.sum(axis=(0, 2, 3))
        sumsq_real += (real_part ** 2).sum(axis=(0, 2, 3))
        sumsq_imag += (imag_part ** 2).sum(axis=(0, 2, 3))
        sum_cross += (real_part * imag_part).sum(axis=(0, 2, 3))
        
        total_pixels += B * H * W
        samples_processed += B
        
        if max_samples and samples_processed >= max_samples:
            break
            
    # Finalize stats
    mean_real = sum_real / total_pixels
    mean_imag = sum_imag / total_pixels
    
    # Var = E[x^2] - (E[x])^2
    var_real = (sumsq_real / total_pixels) - (mean_real ** 2)
    var_imag = (sumsq_imag / total_pixels) - (mean_imag ** 2)
    
    # Covariance cross-term E[xy] - E[x]E[y]
    cov_cross = (sum_cross / total_pixels) - (mean_real * mean_imag)
    
    std_real = np.sqrt(np.maximum(var_real, 0))
    std_imag = np.sqrt(np.maximum(var_imag, 0))
    
    # Construct 2x2 covariance matrices per channel
    # [[Var(R), Cov(R,I)], [Cov(R,I), Var(I)]]
    cov_mats = np.zeros((n_channels, 2, 2))
    cov_mats[:, 0, 0] = var_real
    cov_mats[:, 1, 1] = var_imag
    cov_mats[:, 0, 1] = cov_cross
    cov_mats[:, 1, 0] = cov_cross
    
    # 2D Mean vector
    mean_vecs = np.stack([mean_real, mean_imag], axis=1)

    return {
        "min_real_value": min_vals.tolist(),
        "max_real_value": max_vals.tolist(),
        "mean_real": mean_real.tolist(),
        "mean_imag": mean_imag.tolist(),
        "std_real": std_real.tolist(),
        "std_imag": std_imag.tolist(),
        # For compatibility with legacy config keys
        "mean_real_value": mean_real.tolist(), 
        "std_real_value": std_real.tolist(),
        "mean_complex_value": mean_vecs.tolist(),
        "cov_complex_value": cov_mats.tolist()
    }