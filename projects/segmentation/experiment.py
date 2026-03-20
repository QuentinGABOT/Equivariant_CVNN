"""
Experiment implementation for the Segmentation task.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

# Local imports
from cvnn.base_experiment import BaseExperiment
from cvnn.data import (
    get_dataset_split_indices,
)
from cvnn.evaluate import (
    evaluate,
)
from cvnn.inference import (
    reconstruct_full_segmentation,
)
from cvnn.model_utils import build_model_from_config
from cvnn.plugins import register_plugin
from cvnn.utils import setup_logging
from cvnn.visualize import (
    plot_dataset_split_mask,
    create_dataset_split_mask,
    plot_losses,
    plot_segmentation_full_image,
    plot_classification_metrics,
    plot_segmentations,
)
from cvnn.inference import (
    inference_on_dataloader,
)
from cvnn.data_processing import revert_transforms


# initialize module-level logger
logger = setup_logging(__name__)


@register_plugin("segmentation")
class SegmentationExperiment(BaseExperiment):
    def build_model(self) -> torch.nn.Module:
        """Build UNet model for segmentation using model registry."""
        return build_model_from_config(self.cfg, task="segmentation")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate segmentation performance."""
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
            wandb.log({f"segmentation/{k}": v for k, v in self.metrics["metrics"].items()})
        logger.info(f"Segmentation metrics: {self.metrics}")
        return self.metrics

    def visualize(self) -> None:
        """Create visualizations for segmentation task."""
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

            (
                original_image,
                ground_truth_segmentation,
                predicted_segmentation,
            ) = reconstruct_full_segmentation(
                self.model,
                self.full_loader,
                self.cfg,
                self.device,
                self.nsamples_per_rows,
                self.nsamples_per_cols,
            )
            original_image = revert_transforms(original_image, self.cfg)

            logger.info("Successfully reconstructed full image segmentation")

            fig = plot_segmentation_full_image(
                ground_truth_segmentation=ground_truth_segmentation,
                predicted_segmentation=predicted_segmentation,
                original_image=original_image,
                dataset_type=self.cfg["data"].get("type").lower(),
                number_classes=self.cfg["model"].get("inferred_num_classes"),
                ignore_index=self.cfg["data"].get("ignore_index"),
            )
            save_path = Path(self.logdir) / "full_image_segmentation_analysis.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved full image segmentation analysis to {save_path}")
            if self.wandb_log and hasattr(wandb, "run") and wandb.run:
                wandb.log(
                    {
                        "segmentation/full_image_analysis": wandb.Image(
                            fig, caption="Full Image Segmentation Analysis"
                        )
                    }
                )
            plt.close(fig)

        # Plot training history if available
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
                wandb.log(
                    {
                        "training/loss_curve": wandb.Image(
                            fig, caption="Training & Validation Loss Curve"
                        )
                    }
                )
            plt.close(fig)
        
        fig = plot_classification_metrics(
            self.metrics["metrics"],
            ignore_index=self.cfg["data"].get("ignore_index")
        )
        save_path = Path(self.logdir) / "classification_results.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved classification results to {save_path}")
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            wandb.log({"classification_results": wandb.Image(fig)})
        plt.close(fig)

        ### Show segmentation results for test set ###
        inputs, outputs, targets, _, _, _ = inference_on_dataloader(
            model=self.model,
            data_loader=eval_loader,
            device=self.device,
        )
        inputs = revert_transforms(inputs, self.cfg)
        outputs = torch.argmax(outputs, dim=1).cpu()
        targets = targets.cpu()

        fig = plot_segmentations(
            inputs=inputs,
            predictions=outputs,
            labels=targets,
            number_classes=self.cfg["model"].get("inferred_num_classes"),
            dataset_type=self.cfg["data"].get("type").lower(),
            num_samples=5,
        )
        save_path = Path(self.logdir) / "segmentations.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved segmentations to {save_path}")
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            wandb.log({"segmentations": wandb.Image(fig)})
        plt.close(fig)