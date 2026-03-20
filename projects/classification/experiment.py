"""
Experiment implementation for the Classification task.

"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import torch
import wandb
import matplotlib.pyplot as plt

# Local imports
from cvnn.base_experiment import BaseExperiment
from cvnn.evaluate import (
    evaluate,
)
from cvnn.model_utils import build_model_from_config
from cvnn.plugins import register_plugin
from cvnn.utils import setup_logging
from cvnn.visualize import (
    plot_losses,
    plot_classification_metrics,
    plot_classifications,
)
from cvnn.inference import (
    inference_on_dataloader,
)
from cvnn.data_processing import revert_transforms


# initialize module-level logger
logger = setup_logging(__name__)


@register_plugin("classification")
class ClassificationExperiment(BaseExperiment):
    def build_model(self) -> torch.nn.Module:
        """Build ResNet model for classification using model registry."""
        return build_model_from_config(self.cfg, task="classification")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate classification performance."""
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
            wandb.log({f"classification/{k}": v for k, v in self.metrics["metrics"].items()})
        logger.info(f"Classification metrics: {self.metrics}")
        return self.metrics


    def visualize(self) -> None:
        """Create visualizations for classification task."""
        # Use test_loader if available, otherwise use valid_loader
        eval_loader = (
            self.test_loader if self.test_loader is not None else self.valid_loader
        )

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
                wandb.log({"loss_curve": wandb.Image(fig)})
            plt.close(fig)
        
        fig = plot_classification_metrics(
            self.metrics["metrics"],
        )
        save_path = Path(self.logdir) / "classification_results.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved classification results to {save_path}")
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            wandb.log({"classification_results": wandb.Image(fig)})
        plt.close(fig)


        ### Show classification results for test set ###

        inputs, outputs, targets, _, _, _ = inference_on_dataloader(
            model=self.model,
            data_loader=eval_loader,
            device=self.device,
        )
        inputs = revert_transforms(inputs, self.cfg)
        outputs = torch.argmax(outputs, dim=1)
        fig = plot_classifications(
            inputs=inputs,
            predictions=outputs,
            labels=targets,
            dataset_type=self.cfg["data"].get("type").lower(),
            num_samples=5,
        )
        save_path = Path(self.logdir) / "classifications.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved classifications to {save_path}")
        if self.wandb_log and hasattr(wandb, "run") and wandb.run:
            wandb.log({"classifications": wandb.Image(fig)})
        plt.close(fig)



