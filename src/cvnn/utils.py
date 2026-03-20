# Standard library imports
import logging
import random
from typing import Optional, Union
import time

# Third-party imports
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
import math

def set_seed(seed: int) -> None:
    """
    Set seed for reproducible results across random, numpy, and torch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def format_params(n):
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_params_fmt": format_params(trainable_params),
        "total_params_fmt": format_params(total_params),
    }

def measure_inference_time(model, device, sample_input, warmup_iters=10, timed_iters=50, batch_size=1):# -> dict[str, Any]:
    dtype = sample_input.dtype
    x = torch.randn_like(sample_input, dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(timed_iters):
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "timed_iters": timed_iters,
        "batch_size": batch_size,
        "device": str(device),
    }
    return stats

def estimate_model_flops(model, sample_input, device):
    dtype = sample_input.dtype
    dummy_input = torch.randn_like(sample_input, dtype=dtype, device=device)
    flops = FlopCountAnalysis(model, dummy_input).total()
    def format_flops(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}G"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)
    return {"flops": int(flops), "flops_fmt": format_flops(flops)}

def setup_logging(
    name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Return a logger with StreamHandler and formatter if not already configured.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def safe_log(
    x: Union[float, np.ndarray], base: float = 10, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Compute logarithm safely by avoiding log(0) and negative values.

    Args:
        x: Input value(s) to compute logarithm of
        base: Logarithm base (10 for log10, math.e for natural log)
        eps: Minimum value to clamp inputs to avoid numerical issues

    Returns:
        Logarithm of the input with the specified base

    Raises:
        ValueError: If base is invalid
    """
    if base <= 0 or base == 1:
        raise ValueError(f"Invalid logarithm base: {base}")

    x_safe = np.clip(x, eps, None)
    if base == 10:
        return np.log10(x_safe)
    elif base == math.e:
        return np.log(x_safe)
    else:
        # Change of base formula: log_b(x) = ln(x) / ln(b)
        return np.log(x_safe) / np.log(base)

@torch.no_grad()
def kl_sanity_checks(device="cuda" if torch.cuda.is_available() else "cpu", D=64):
    """
    Verifies:
      (A) Prior case: mu=0, std=1, delta=0 -> KL=0
      (B) δ=0 case matches closed form: (σ² + |μ|² - 1) - log(σ²)
      (C) Monotonicity near boundary as |δ|↑
    """
    def kl_impl(mu, std, delta, eps=1e-12):
        sigma2 = std**2
        t = (sigma2**2 - delta.abs()**2).clamp_min(eps)
        per_dim = (sigma2 + mu.abs()**2 - 1.0) - 0.5 * torch.log(t)
        return per_dim.view(mu.size(0), -1).sum(1)  # per-sample

    B = 4
    # (A) prior
    mu = torch.zeros(B, D, dtype=torch.complex64, device=device)
    std = torch.ones(B, D, dtype=torch.float32, device=device)
    delta = torch.zeros(B, D, dtype=torch.complex64, device=device)
    KL = kl_impl(mu, std, delta)
    assert KL.abs().max().item() < 1e-6, f"Prior KL not zero: {KL}"

    # (B) δ=0 closed-form
    torch.manual_seed(0)
    mu = (torch.randn(B, D, device=device) + 1j*torch.randn(B, D, device=device)) * 0.1
    std = torch.exp(torch.randn(B, D, device=device) * 0.1)  # positive
    delta = torch.zeros(B, D, dtype=torch.complex64, device=device)
    KL1 = kl_impl(mu, std, delta)
    KL2 = ((std**2 + mu.abs()**2 - 1.0) - torch.log(std**2)).sum(1)
    assert torch.allclose(KL1, KL2, rtol=1e-5, atol=1e-6), f"δ=0 mismatch: {(KL1-KL2).abs().max()}"

    # (C) boundary monotonicity: increase |δ| -> KL increases
    phi = torch.randn(B, D, device=device)
    phase = torch.exp(1j * phi)
    sigma2 = std**2
    deltas = [0.0, 0.5, 0.9, 0.99]
    KLs = []
    for alpha in deltas:
        delta = (alpha := alpha) * sigma2 * phase * 0.999  # keep strictly inside feasible set
        KLs.append(kl_impl(mu, std, delta).mean().item())
    assert KLs == sorted(KLs), f"KL not monotone with |δ|: {KLs}"
    print("[KL sanity] passed:", {"prior": KL.mean().item(), "delta0_eq": (KL1-KL2).abs().max().item(), "monotone": KLs})

class KLAnnealing:
    def __init__(self, kind, warmup_steps):
        self.kind = kind
        self.N = warmup_steps
        self.t = 0

    def progress(self) -> float:
        """
        Returns a scalar in [0,1] indicating how far along the schedule we are.
        """
        # --- NOUVEAU MODE ICI ---
        if self.kind == "constant" or self.kind is None:
            return 1.0
        
        if self.N <= 0:
            return 1.0

        x = min(1.0, self.t / self.N)
        
        if self.kind == "linear":
            return x
        if self.kind == "cosine":
            return 0.5 * (1 - math.cos(math.pi * x))
        if self.kind == "sigmoid":
            s = 10.0
            return 1 / (1 + math.exp(-s * (x - 0.5)))
            
        raise ValueError(f"Unknown schedule kind: {self.kind}")

    def step(self):
        """Advance the schedule by one step (usually one minibatch)."""
        self.t += 1