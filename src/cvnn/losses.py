"""
Custom loss functions for CVNN models.

This module provides custom loss implementations for both real and complex-valued
neural networks, including advanced losses like Focal Loss with class weighting.
"""

from typing import Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn

def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    weight_mode: str = "inverse_frequency",
    ignore_index: int = -100,
) -> torch.Tensor:
    if device is None:
        device = targets.device

    # Mask out ignore_index
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        mask = targets != ignore_index
        valid = targets[mask]
    else:
        valid = targets

    # Count frequencies (produce float32 to avoid unexpected double dtype)
    counts = (
        torch.bincount(valid.flatten(), minlength=num_classes)
        .to(dtype=torch.float32)
        .clamp(min=1.0)
    )
    # Zero out ignored class so it won't contribute
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        counts[ignore_index] = float("inf")

    if weight_mode == "inverse_frequency":
        w = 1.0 / counts
    elif weight_mode == "balanced":
        N = valid.numel()
        # normalize by true number of classes present (exclude ignore)
        C = num_classes - (
            1 if ignore_index is not None and 0 <= ignore_index < num_classes else 0
        )
        w = N / (C * counts)
    elif weight_mode == "log_frequency":
        w = torch.log1p(1.0 / counts)
    else:
        raise ValueError(f"Unknown weight_mode {weight_mode!r}")

    # Zero out ignored class weight
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        w[ignore_index] = 0.0

    # Normalize so sum(w) == num_classes (or num_classes-1)
    norm_C = num_classes - (
        1 if ignore_index is not None and 0 <= ignore_index < num_classes else 0
    )
    w = w * norm_C / w.sum()

    # make sure returned weights are float32 on the requested device
    return w.to(dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        use_class_weights: bool = True,
        weight_mode: str = "balanced",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight)  # so it moves with .to(device)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_class_weights = use_class_weights
        self.weight_mode = weight_mode

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N, C = inputs.shape[:2]
        # compute or reuse class‐weights for CE
        if self.use_class_weights and self.weight is None:
            class_w = compute_class_weights(
                targets,
                C,
                device=inputs.device,
                weight_mode=self.weight_mode,
                ignore_index=self.ignore_index,
            )
        else:
            class_w = self.weight

        # flatten spatial dims if present
        # flatten spatial dims if present and ensure logits are float32
        logits = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, C)
        if logits.dtype != torch.float32 and logits.is_floating_point():
            logits = logits.to(dtype=torch.float32)

        # targets must be integer class labels (long)
        t = targets.view(-1).long()
        valid = t != self.ignore_index

        # compute log‐probs and probs
        logpt = F.log_softmax(logits, dim=1)  # [M, C]
        pt = logpt.exp()

        # pick the log‐prob of the true class
        logpt = logpt[valid, t[valid]]
        pt = pt[valid, t[valid]]

        # alpha per sample
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[t[valid]]
        else:
            alpha_t = self.alpha

        # focal loss
        focal = alpha_t * (1 - pt) ** self.gamma * (-logpt)

        # apply class‐weights inside CE if available
        if class_w is not None:
            # ensure weight dtype matches logits (float32)
            if class_w.dtype != torch.float32 and class_w.is_floating_point():
                class_w = class_w.to(dtype=torch.float32)

            ce = F.cross_entropy(
                logits,
                t,
                weight=class_w,
                ignore_index=self.ignore_index,
                reduction="none",
            )[valid]
            # Replace our focal CE term with weighted CE version:
            focal = alpha_t * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal