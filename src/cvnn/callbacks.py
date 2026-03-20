"""
cvnn.callbacks

This module contains training callbacks such as ModelCheckpoint.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Optional, Union

# Third-party imports
import torch


class ModelCheckpoint:
    """
    Early stopping and checkpointing callback for training.

    Saves best and last model states based on validation metrics,
    including optimizer and scheduler state dictionaries for resuming training.

    Args:
        model: PyTorch model to checkpoint
        optimizer: Optimizer being used for training
        savepath: Directory path to save checkpoint files
        num_input_dims: Number of input dimensions for the model
        min_is_best: If True, lower scores are better (e.g., loss)
        warmup_scheduler: Optional warmup scheduler to save
        scheduler: Optional main scheduler to save
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        savepath: Union[str, Path],
        num_input_dims: int,
        min_is_best: bool = True,
        warmup_scheduler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.savepath = Path(savepath)
        self.num_input_dims = num_input_dims
        self.warmup_scheduler = warmup_scheduler
        self.scheduler = scheduler
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score: float) -> bool:
        """Check if current score is better when lower is better."""
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score: float) -> bool:
        """Check if current score is better when higher is better."""
        return self.best_score is None or score > self.best_score

    def update(self, score: float, epoch: int) -> bool:
        """
        Update checkpoint if score improved.

        Args:
            score: Current validation score
            epoch: Current training epoch

        Returns:
            True if checkpoint was saved (score improved), False otherwise
        """
        if self.is_better(score):
            self.model.eval()
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": score,
            }
            if self.warmup_scheduler is not None:
                state["warmup_state_dict"] = self.warmup_scheduler.state_dict()
            if self.scheduler is not None:
                state["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(state, self.savepath / "best_model.pt")
            self.best_score = score
            return True
        return False
