import torch
import torch.nn as nn
from torchcvnn.nn.modules.loss import ComplexMSELoss
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ComplexMeanSquareError(nn.Module):
    def __init__(self):
        super(ComplexMeanSquareError, self).__init__()

    def forward(self, y_true, y_pred):

        # Calculate Mean Square Error
        mse = torch.mean(torch.square(torch.abs(y_true - y_pred)))

        return mse


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, ignore_index=-100, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, softmax_probs, targets):
        """
        Args:
            softmax_probs: Tensor of precomputed softmax probabilities (already averaged if necessary).
            targets: Ground truth labels.
        """
        # Safeguard to prevent log(0)
        softmax_probs = torch.clamp(softmax_probs, min=1e-10, max=1.0)

        # Convert to log probabilities
        log_probs = torch.log(softmax_probs)

        # Convert targets to int64 type
        targets = targets.type(torch.int64)

        # Compute cross-entropy loss (negative log-likelihood loss)
        ce_loss = F.nll_loss(
            log_probs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Compute probabilities of the true class
        pt = torch.exp(-ce_loss)

        # Apply weighting with alpha and focusing with gamma
        focal_term = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            loss = (alpha_weights * focal_term * ce_loss).mean()
        else:
            loss = (focal_term * ce_loss).mean()

        return loss


class ComplexCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights, ignore_index=0):
        super(ComplexCrossEntropyLoss, self).__init__()
        self.weight = class_weights
        self.ignore_index = ignore_index

    def forward(self, mean_softmax, targets):
        targets = targets.type(torch.int64)
        loss = torch.mean(
            F.cross_entropy(
                mean_softmax,
                targets,
                ignore_index=self.ignore_index,
                weight=self.weight,
            )
        )
        return loss
