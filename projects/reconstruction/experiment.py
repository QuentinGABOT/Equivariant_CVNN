"""
Experiment implementation for the Reconstruction task, separated from core pipeline.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from cvnn.base_experiment import BaseExperiment
from cvnn.model_utils import build_model_from_config
from cvnn.plugins import register_plugin
from cvnn.utils import setup_logging
from cvnn.data import (
    get_dataset_split_indices,
)
from cvnn.evaluate import (
    evaluate,
)
from cvnn.inference import (
    reconstruct_full_image,
    inference_on_dataloader,
)
from cvnn.visualize import (
    create_dataset_split_mask,
    plot_dataset_split_mask,
    plot_losses,
    plot_reconstructions,
    plot_pauli_decomposition,
    plot_krogager_decomposition,
    plot_cameron_decomposition,
    plot_h_alpha_decomposition,
    plot_h_alpha_plane,
    plot_reconstruction_error_analysis,
    plot_classification_metrics,
)
from cvnn.data_processing import revert_transforms
from cvnn.metrics_registry import (
    compute_h_alpha_metrics,
    compute_cameron_metrics,
    compute_reconstruction_errors,
)

# initialize module-level logger
logger = setup_logging(__name__)


@register_plugin("reconstruction")
class ReconstructionExperiment(BaseExperiment):
    """Reconstruction experiment implementation.

    This class handles training, evaluation, and visualization for
    reconstruction tasks using AutoEncoder architecture.
    """

    def build_model(self) -> torch.nn.Module:
        """Build autoencoder model for reconstruction task using model registry.
        
        Returns:
            torch.nn.Module: The autoencoder model for reconstruction.
        """
        return build_model_from_config(self.cfg, task="reconstruction")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate reconstruction quality.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        # Use test_loader if available, otherwise use valid_loader
        eval_loader = (
            self.test_loader if self.test_loader is not None else self.valid_loader
        )

        self.metrics = evaluate(
            task=self.cfg.get("task"),  # "reconstruction", "generation", etc.
            model=self.model,
            test_loader=eval_loader,
            cfg=self.cfg,
            device=self.device
        )
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            if "stats" in self.metrics:
                wandb.log({f"stats/{k}": v for k, v in self.metrics["stats"].items()})
            if "consistency" in self.metrics:
                wandb.log({f"consistency/{k}": v for k, v in self.metrics["consistency"].items()})
            wandb.log({f"reconstruction/{k}": v for k, v in self.metrics["metrics"].items()})
        logger.info(f"Reconstruction metrics: {self.metrics}")

        if self.cfg["data"].get(
            "supports_full_image_reconstruction"
        ):
            # Reconstruct full image for physics-based metrics (and visualizations)
            original_image, reconstruct_image = reconstruct_full_image(
                self.model,
                self.full_loader,
                config=self.cfg,
                device=self.device,
                nsamples_per_rows=self.nsamples_per_rows,
                nsamples_per_cols=self.nsamples_per_cols,
            )
            original_image = revert_transforms(original_image, self.cfg)
            reconstruct_image = revert_transforms(reconstruct_image, self.cfg)

            reconstruction_errors = compute_reconstruction_errors(
                original=original_image,
                reconstructed=reconstruct_image,
                cfg=self.cfg,
            )
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                for error_name, error_values in reconstruction_errors.items():
                    if error_values.size > 0:
                        wandb.log({f"metrics/reconstruction_errors/{error_name}_mean": np.mean(error_values)})
                        wandb.log({f"metrics/reconstruction_errors/{error_name}_std": np.std(error_values)})
            self.metrics["error_reconstruction"] = reconstruction_errors
            
            if self.cfg["data"].get("type").lower() == "polsar":
                h_alpha_metrics = compute_h_alpha_metrics(
                    image1=original_image,
                    image2=reconstruct_image,
                )
                if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                    wandb.log({f"metrics/h_alpha/{k}": v for k, v in h_alpha_metrics.items() 
                               if k not in ["per_class_metrics", "confusion_matrix_raw", "confusion_matrix_normalized", "class_labels"]})
                    for class_id, class_metrics in h_alpha_metrics["per_class_metrics"].items():
                        for metric_name, metric_value in class_metrics.items():
                            wandb.log({f"metrics/h_alpha/class_{class_id}/{metric_name}": metric_value})
                self.metrics["h_alpha"] = h_alpha_metrics

                cameron_metrics = compute_cameron_metrics(
                    image1=original_image,
                    image2=reconstruct_image,
                )
                if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                    wandb.log({f"metrics/cameron/{k}": v for k, v in cameron_metrics.items() 
                               if k not in ["per_class_metrics", "confusion_matrix_raw", "confusion_matrix_normalized", "class_labels"]})
                    for class_id, class_metrics in cameron_metrics["per_class_metrics"].items():
                        for metric_name, metric_value in class_metrics.items():
                            wandb.log({f"metrics/cameron/class_{class_id}/{metric_name}": metric_value})
                self.metrics["cameron"] = cameron_metrics

            self.original_image = original_image
            self.reconstruct_image = reconstruct_image
        else:
            self.original_image = None
            self.reconstruct_image = None

        return self.metrics

    def visualize(self) -> None:
        # Use test_loader if available, otherwise use valid_loader
        eval_loader = (
            self.test_loader if self.test_loader is not None else self.valid_loader
        )

        if self.cfg["data"].get(
            "supports_full_image_reconstruction"
        ):
            # Create dataset split visualization for patch-based datasets
            train_indices, valid_indices, test_indices = get_dataset_split_indices(self.cfg)

            mask = create_dataset_split_mask(
                cfg=self.cfg,
                full_loader=self.full_loader,
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices,
                nsamples_per_cols=self.nsamples_per_cols,
                nsamples_per_rows=self.nsamples_per_rows,
            )
            fig = plot_dataset_split_mask(
                mask=mask,
                patch_size=self.cfg["data"]["dataset"]["patch_size"],
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices,
            )
            save_path = Path(self.logdir) / "dataset_split_visualization.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved dataset split visualization to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log(
                    {
                        "dataset_split/visualization": wandb.Image(fig, caption="Dataset Split Visualization"),
                        "dataset_split/train_patches": len(train_indices),
                        "dataset_split/valid_patches": len(valid_indices),
                        "dataset_split/test_patches": len(test_indices) if test_indices else 0,
                        "dataset_split/patch_size": self.cfg["data"]["dataset"]["patch_size"],
                        "dataset_split/total_patches": len(train_indices)
                        + len(valid_indices)
                        + (len(test_indices) if test_indices else 0),
                    }
                )
            plt.close(fig)

            fig = plot_reconstruction_error_analysis(self.metrics["error_reconstruction"])
            save_path = Path(self.logdir) / "reconstruction_error_analysis.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved reconstruction error analysis to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"reconstruction_error_analysis": wandb.Image(fig)})
            plt.close(fig)

        if self.cfg.get("mode") in ("full", "train", "retrain"):
            assert (
                self.history is not None
            ), "`train()` must be called before `visualize()`"
            fig = plot_losses(
                self.history["train_loss"],
                self.history["valid_loss"],
            )
            save_path = Path(self.logdir) / "loss_curve.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved loss curve to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"loss_curve": wandb.Image(fig)})
            plt.close(fig)

        if self.cfg["data"].get("type").lower() == "polsar":
            fig = plot_pauli_decomposition(
                self.original_image,
                self.reconstruct_image,
            )
            save_path = Path(self.logdir) / "pauli_decomposition.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Pauli decomposition to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"pauli_decomposition": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_krogager_decomposition(
                self.original_image,
                self.reconstruct_image,
            )
            save_path = Path(self.logdir) / "krogager_decomposition.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Krogager decomposition to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"krogager_decomposition": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_h_alpha_decomposition(
                self.original_image,
                self.reconstruct_image,
            )
            save_path = Path(self.logdir) / "h_alpha_decomposition.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved H/A decomposition to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"h_alpha_decomposition": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_h_alpha_plane(
                self.original_image,
                self.reconstruct_image,
            )
            save_path = Path(self.logdir) / "h_alpha_plane.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved H/Alpha plane to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"h_alpha_plane": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_classification_metrics(
                self.metrics["h_alpha"],
                class_names= {
                    1: "Complex structures",
                    2: "Random anisotropic scatterers",
                    4: "Double reflection propagation effects",
                    5: "Anisotropic particles",
                    6: "Random surfaces",
                    7: "Dihedral reflector",
                    8: "Dipole",
                    9: "Bragg surface",
                }
            )
            save_path = Path(self.logdir) / "h_alpha_classification_metrics.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved H/Alpha classification metrics to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"h_alpha_classification_metrics": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_cameron_decomposition(
                self.original_image,
                self.reconstruct_image,
            )
            save_path = Path(self.logdir) / "cameron_decomposition.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Cameron decomposition to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"cameron_decomposition": wandb.Image(fig)})
            plt.close(fig)

            fig = plot_classification_metrics(
                self.metrics["cameron"],
                class_names = {
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
            )
            save_path = Path(self.logdir) / "cameron_classification_metrics.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Cameron classification metrics to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log({"cameron_classification_metrics": wandb.Image(fig)})
            plt.close(fig)

        ### Show reconstructions for test set ###

        inputs, outputs, _, _, _, _ = inference_on_dataloader(
            model=self.model,
            data_loader=eval_loader,
            device=self.device,
        )
        inputs = revert_transforms(inputs, self.cfg)
        outputs = revert_transforms(outputs, self.cfg)
        fig = plot_reconstructions(
            inputs=inputs,
            outputs=outputs,
            dataset_type=self.cfg["data"].get("type").lower(),
            num_samples=5,
        )
        save_path = Path(self.logdir) / "reconstructions.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reconstructions to {save_path}")
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            wandb.log({"reconstructions": wandb.Image(fig)})
        plt.close(fig)
