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


class ComplexAmplitudePhaseError(nn.Module):
    def __init__(self):
        super(ComplexAmplitudePhaseError, self).__init__()

    def forward(self, y_true, y_pred):

        # Calculate Mean Square Error
        MSELoss = ComplexMSELoss()
        mse = MSELoss(torch.abs(y_true), torch.abs(y_pred))
        mean_phase_error = torch.mean(
            torch.abs(y_true)
            * (1 - torch.cos(torch.angle(y_true) - torch.angle(y_pred)))
        )

        return mse + mean_phase_error


class ComplexHuberLoss(nn.Module):
    def __init__(self):
        super(ComplexHuberLoss, self).__init__()

    def forward(self, y_true, y_pred, delta=1.0):

        # Calculate Huber Loss
        l1 = torch.abs(y_true - y_pred)
        if l1 < delta:
            huber = delta * (torch.abs(y_true - y_pred) ** 2)
        else:
            huber = torch.mean(delta * (torch.abs(y_true - y_pred) - 0.5 * delta))
        return huber


class ComplexVAELoss(nn.Module):
    def __init__(self):
        """
        Initializes the VAE Loss module.

        """
        super(ComplexVAELoss, self).__init__()

    def forward(self, x, recon_x, mu, sigma, delta, kld_weight):
        """
        Computes the VAE loss.

        Parameters:
        recon_x: Reconstructed data.
        x: Original input data.
        mu: Mean from the latent space.
        sigma: Covariance from the latent space.
        delta: Pseudo covariance from the latent space.

        Returns:
        torch.Tensor: Computed VAE loss.
        """
        # Reconstruction Loss-
        MSELoss = ComplexMSELoss()
        recon_loss = MSELoss(y_true=x, y_pred=recon_x)

        # KL Divergence
        kl_divergence = (
            -mu.shape[1]
            + torch.sum(
                (
                    sigma * (1 + torch.square(torch.abs(mu)))
                    + (delta * torch.square(1j * mu)).real
                )
                / (torch.square(sigma) - torch.square(torch.abs(delta)))
                + 0.5 * torch.log(torch.square(sigma) - torch.square(torch.abs(delta)))
            )
        ) / (mu.shape[1] * mu.shape[0])

        return (
            recon_loss + kld_weight * kl_divergence,
            recon_loss,
            kl_divergence,
            torch.mean(mu),
            torch.mean(sigma),
            torch.mean(delta),
        )


class ComplexVAEPhaseLoss(nn.Module):
    def __init__(self):
        """
        Initializes the VAE Loss module.

        """
        super(ComplexVAEPhaseLoss, self).__init__()

    def forward(self, x, recon_x, mu, sigma, delta, kld_weight):
        """
        Computes the VAE loss.

        Parameters:
        recon_x: Reconstructed data.
        x: Original input data.
        mu: Mean from the latent space.
        sigma: Covariance from the latent space.
        delta: Pseudo covariance from the latent space.

        Returns:
        torch.Tensor: Computed VAE loss.
        """
        # Reconstruction Loss-
        MSELoss = ComplexAmplitudePhaseError()
        recon_loss = MSELoss(y_true=x, y_pred=recon_x)

        # KL Divergence
        kl_divergence = (
            -mu.shape[1]
            + torch.sum(
                (
                    (
                        sigma * (1 + torch.square(torch.abs(mu)))
                        + (delta * torch.square(1j * mu)).real
                    )
                    / (torch.square(sigma) - torch.square(torch.abs(delta)))
                    + 0.5
                    * torch.log(torch.square(sigma) - torch.square(torch.abs(delta)))
                )
            )
        ) / (mu.shape[1] * mu.shape[0])

        return recon_loss + kld_weight * kl_divergence, recon_loss, kl_divergence


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
