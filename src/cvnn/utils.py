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