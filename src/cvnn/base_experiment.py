"""
Base experiment class for CVNN tasks.
"""

# Standard library imports
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torchinfo
import wandb
import yaml

# Local imports
from cvnn.config import load_config
from cvnn.data import (
    get_dataloaders,
    get_full_image_dataloader,
)
from cvnn.models.utils import init_weights_mode_aware
from cvnn.schedulers import build_schedulers
from cvnn.train import setup_loss_optimizer, train_model
from cvnn.utils import set_seed, setup_logging
from cvnn.wandb_utils import setup_wandb

# initialize module-level logger
logger = setup_logging(__name__)


class BaseExperiment(ABC):
    """
    Base class for all CVNN experiments.

    Provides common functionality for experiment setup, training, evaluation,
    and visualization across different tasks (reconstruction, segmentation, etc.).

    Args:
        config_path: Path to YAML configuration file
        resume_logdir: Optional path to existing log directory for resuming
        mode_override: Optional mode override ('train', 'eval', 'full')
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        resume_logdir: Optional[Union[str, Path]] = None,
        mode_override: Optional[str] = None,
    ) -> None:
        """Initialize experiment: load config, set up logging, seeds, data, model, and summaries."""
        # load config
        self.cfg = load_config(str(config_path))
        # apply mode override if provided
        if mode_override is not None:
            self.cfg["mode"] = mode_override
        # setup wandb logging and run name
        self.wandb_log, self.run_name = setup_wandb(self.cfg, resume_logdir)
        # seed
        seed = self.cfg.get("seed") or random.randint(0, 9999)
        self.cfg["seed"] = seed
        set_seed(seed)
        logger.info(f"Using seed: {seed}")
        # versions
        logger.info(f"Python: {sys.version}")
        logger.info(f"NumPy: {np.__version__}")
        logger.info(f"PyTorch: {torch.__version__}")
        try:
            git_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
            logger.info(f"Git SHA: {git_sha}")
        except Exception:
            pass
        # device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f"Using device: {self.device}")
        # determine log directory (new or existing)
        if resume_logdir:
            # resume_logdir may be str or Path
            self.logdir = Path(resume_logdir)
            if not self.logdir.exists():
                raise FileNotFoundError(f"Resume logdir not found: {self.logdir}")
            # use existing folder
            self.cfg["logging"]["logdir"] = str(self.logdir)
            with open(self.logdir / "config.yml", "r") as f:
                self.cfg = yaml.safe_load(f)
            # Re-apply mode override after loading config (important for retrain!)
            if mode_override is not None:
                self.cfg["mode"] = mode_override
        else:
            root = self.cfg["logging"]["logdir"]
            task = self.cfg.get("task", "experiment")
            self.logdir = Path(root) / (
                f"{task}_{self.run_name}" if self.run_name else task
            )
            self.logdir.mkdir(parents=True, exist_ok=True)
        # data loaders (train/valid and optional test)
        loaders = get_dataloaders(self.cfg, self.use_cuda)

        with open(self.logdir / "config.yml", "w") as f:
            try:
                # Try to dump the config, but handle non-serializable objects
                yaml.dump(self.cfg, f, default_flow_style=False)
            except (TypeError, ValueError) as e:
                # If direct dump fails, create a serializable version
                logger.warning(f"Config contains non-serializable objects: {e}")
                serializable_cfg = self._make_serializable(self.cfg)
                yaml.dump(serializable_cfg, f, default_flow_style=False)

        if len(loaders) == 3:
            self.train_loader, self.valid_loader, self.test_loader = loaders
        else:
            self.train_loader, self.valid_loader = loaders  # type: ignore
            self.test_loader = None
        # Build full-image dataloader only for datasets that support it
        dataset_name = self.cfg["data"]["dataset"]["name"]

        if self.cfg["data"]["supports_full_image_reconstruction"]:
            (
                self.full_loader,
                self.nsamples_per_rows,
                self.nsamples_per_cols,
            ) = get_full_image_dataloader(self.cfg, self.use_cuda)
        else:
            # Some datasets (e.g. Sethi) don't support full-image reconstruction.
            self.full_loader = None
            self.nsamples_per_rows = 0
            self.nsamples_per_cols = 0
            logger.info(
                f"Dataset {dataset_name} does not support full-image reconstruction; skipping full image dataloader"
            )
        # model
        self.model = self.build_model()

        # setup gumbel tau if applicable
        self._setup_gumbel_tau()

        # setup projection state management if applicable
        self._setup_projection_state()

        summary_text = (
            f"Logdir: {self.logdir}\n"
            + "## Command\n"
            + f"{' '.join(sys.argv)}\n\n"
            + f"Config: {self.cfg}\n\n"
            + (f"Wandb run name: {self.run_name}\n\n" if self.run_name else "")
            + "## Summary of the model architecture\n"
            + f"{torchinfo.summary(self.model)}\n\n"
            + f"{self.model}\n\n"
            + "## Datasets:\n"
            + f"Train: {self.train_loader.dataset}\n"
            + f"Validation: {self.valid_loader.dataset}"
            + (f"\nTest: {self.test_loader.dataset}" if self.test_loader else "")
            + (
                f"\nFull image: {self.full_loader.dataset}\n"
                if self.full_loader is not None
                else "\nFull image: None\n"
            )
        )

        with open(self.logdir / "summary.txt", "w", encoding="utf-8") as file:
            file.write(summary_text)

        logger.info(summary_text)
        if self.cfg.get("wandb"):
            wandb.log({"summary": summary_text})

        # Use mode-aware weight initialization
        layer_mode = self.cfg["model"].get("layer_mode")

        self.model.apply(lambda m: init_weights_mode_aware(m, layer_mode))
        self.model.to(self.device)
        # setup loss and optimizer (for retraining resume)
        self.loss_fn, self.optimizer = setup_loss_optimizer(
            self.model, self.cfg, self.train_loader.dataset, self.device
        )
        # build warmup and main schedulers
        self.warmup_scheduler, self.scheduler = build_schedulers(
            self.optimizer, self.cfg, len(self.train_loader)
        )
        # placeholders
        self.history = None
        self.results = None

    def _setup_gumbel_tau(self) -> None:
        """Initialize Gumbel tau parameter from config if model uses LPD downsampling."""
        # Check if model uses LPD downsampling
        downsampling_layer = self.cfg["model"].get("downsampling_layer")
        if downsampling_layer not in ["LPD", "LPD_F"]:
            self.gumbel_tau_config = None
            return

        # Check if gumbel_tau config exists
        gumbel_config = self.cfg["model"].get("gumbel_tau")
        if gumbel_config is None:
            logger.warning("Model uses LPD but no gumbel_tau config found")
            self.gumbel_tau_config = None
            return

        # Store gumbel tau parameters
        self.gumbel_tau_config = gumbel_config
        self.current_gumbel_tau = torch.tensor(
            gumbel_config["start_value"], dtype=torch.float32
        )

        # Initialize tau on model
        self._initialize_gumbel_tau(self.current_gumbel_tau)
        logger.info(f"Initialized model gumbel_tau to {self.current_gumbel_tau.item()}")

    def _initialize_gumbel_tau(self, tau: torch.Tensor) -> None:
        """Initialize the gumbel tau value for the model encoder blocks."""
        if not hasattr(self.model, "encoder"):
            logger.warning(
                "Model does not have encoder attribute for Gumbel tau initialization"
            )
            return

        for enc in self.model.convnet.encoder[1:]:
            if hasattr(enc, "down") and hasattr(enc.down, "component_selection"):
                enc.down.component_selection.gumbel_tau = tau
            else:
                logger.warning(
                    f"Encoder block missing expected structure for Gumbel tau: {enc}"
                )

    def get_gumbel_tau_state(self) -> Optional[Dict[str, Any]]:
        """Get current Gumbel tau state for checkpoint saving."""
        if self.gumbel_tau_config is None:
            return None
        return {
            "current_gumbel_tau": self.current_gumbel_tau,
            "gumbel_tau_config": self.gumbel_tau_config,
        }

    def load_gumbel_tau_state(self, state: Dict[str, Any]) -> None:
        """Load Gumbel tau state from checkpoint."""
        if state is None:
            return
        self.current_gumbel_tau = state["current_gumbel_tau"]
        self.gumbel_tau_config = state["gumbel_tau_config"]
        self._initialize_gumbel_tau(self.current_gumbel_tau)
        logger.info(f"Restored gumbel_tau to {self.current_gumbel_tau.item()}")

    def _setup_projection_state(self) -> None:
        """Initialize projection state management for polynomial and MLP projections."""
        projection_layer = self.cfg.get("model", {}).get("projection_layer")
        if projection_layer not in ["polynomial", "MLP"]:
            self.projection_config = None
            return

        # Get projection config
        projection_config = self.cfg.get("model", {}).get("projection", {})

        # Set defaults based on projection type
        if projection_layer == "polynomial":
            projection_config.setdefault("order", 3)
        elif projection_layer == "MLP":
            projection_config.setdefault("hidden_sizes", [8, 16])
            projection_config.setdefault("input_size", 2)
            projection_config.setdefault("output_size", 1)

        self.projection_config = {"type": projection_layer, "config": projection_config}

        logger.info(
            f"Initialized projection state management: {projection_layer} with config {projection_config}"
        )

    def get_projection_state(self) -> Optional[Dict[str, Any]]:
        """Get current projection state for checkpoint saving."""
        if self.projection_config is None:
            return None

        # Extract projection layers from model
        projection_layers = {}

        def extract_projections(module, name=""):
            """Recursively extract projection layers from model."""
            from cvnn.models.projection import PolyCtoR, MLPCtoR

            if isinstance(module, (PolyCtoR, MLPCtoR)):
                projection_layers[name] = module.state_dict()

            for child_name, child in module.named_children():
                child_path = f"{name}.{child_name}" if name else child_name
                extract_projections(child, child_path)

        extract_projections(self.model)

        return {
            "projection_config": self.projection_config,
            "projection_layers": projection_layers,
        }

    def load_projection_state(self, state: Dict[str, Any]) -> None:
        """Load projection state from checkpoint."""
        if state is None:
            return

        self.projection_config = state["projection_config"]
        projection_layers = state["projection_layers"]

        def restore_projections(module, name=""):
            """Recursively restore projection layers in model."""
            from cvnn.models.projection import PolyCtoR, MLPCtoR

            if isinstance(module, (PolyCtoR, MLPCtoR)) and name in projection_layers:
                module.load_state_dict(projection_layers[name])
                logger.info(f"Restored projection layer: {name}")

            for child_name, child in module.named_children():
                child_path = f"{name}.{child_name}" if name else child_name
                restore_projections(child, child_path)

        restore_projections(self.model)
        logger.info(f"Restored projection state: {self.projection_config['type']}")

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        pass

    def train(self, start_epoch: int = 0) -> Dict[str, Any]:
        """Run training loop and return history of loss values.

        Args:
            start_epoch: Starting epoch for training (0 for new training, >0 for retrain)
        """
        # Pass self as gumbel_experiment if we have gumbel_tau_config
        gumbel_experiment = self if self.gumbel_tau_config is not None else None

        self.history = train_model(
            self.model,
            self.train_loader,
            self.valid_loader,
            cfg=self.cfg,
            logdir=self.logdir,
            device=self.device,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            warmup_scheduler=self.warmup_scheduler,
            scheduler=self.scheduler,
            start_epoch=start_epoch,
            gumbel_experiment=gumbel_experiment,
        )
        return self.history

    def load_best_model(self) -> None:
        """Load the best model checkpoint saved by ModelCheckpoint."""
        checkpoint_path = self.logdir / "best_model.pt"
        logger.info(f"Loading best model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Load Gumbel tau state if present
        if "gumbel_tau_state" in ckpt:
            self.load_gumbel_tau_state(ckpt["gumbel_tau_state"])

        # Load projection state if present
        if "projection_state" in ckpt:
            self.load_projection_state(ckpt["projection_state"])

    def load_last_model(self) -> int:
        """Load the last model checkpoint (last_model.pt) for retraining.

        Returns:
            int: The epoch number from the checkpoint (to resume from epoch + 1)
        """
        last_path = self.logdir / "last_model.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"No last_model.pt found in {self.logdir}")
        logger.info(f"Loading last checkpoint from {last_path}")
        ckpt = torch.load(last_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        # restore optimizer and scheduler states
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.warmup_scheduler and "warmup_state_dict" in ckpt:
            self.warmup_scheduler.load_state_dict(ckpt["warmup_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Load Gumbel tau state if present
        if "gumbel_tau_state" in ckpt:
            self.load_gumbel_tau_state(ckpt["gumbel_tau_state"])

        # Load projection state if present
        if "projection_state" in ckpt:
            self.load_projection_state(ckpt["projection_state"])

        checkpoint_epoch = ckpt.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {checkpoint_epoch}")
        return checkpoint_epoch

    @abstractmethod
    def evaluate(self) -> Any:
        """Task-specific evaluation; must be implemented by subclass."""
        pass

    @abstractmethod
    def visualize(self) -> None:
        """Task-specific visualization; must be implemented by subclass."""
        pass

    def _make_serializable(self, obj):
        """Convert objects to YAML-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # For objects with attributes, convert to string representation
            return str(obj)
        else:
            # For other objects, try to convert to basic types
            try:
                # Test if it's JSON serializable (which means YAML serializable)
                import json

                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def run(self) -> None:
        """Orchestrate training, evaluation, and visualization based on mode."""
        mode = self.cfg.get("mode")
        # full/train or retrain modes
        if mode in ("full", "train", None):
            logger.info("Starting full training")
            self.train()
        elif mode == "retrain":
            logger.info("Retraining from last checkpoint")
            checkpoint_epoch = self.load_last_model()
            start_epoch = checkpoint_epoch + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
            self.train(start_epoch=start_epoch)
        # evaluation-only path
        if mode in ("full", "train", "retrain"):
            logger.info("Loading best checkpoint for evaluation")
            self.load_best_model()
            logger.info("Evaluating results")
            self.metrics = self.evaluate()
            logger.info("Creating visualizations")
            self.visualize()
        elif mode == "eval":
            logger.info("Evaluation-only mode: loading best checkpoint")
            self.model.eval()
            self.load_best_model()
            logger.info("Evaluating results")
            self.metrics = self.evaluate()
            logger.info("Creating visualizations")
            self.visualize()
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose from 'full', 'train', 'retrain', 'eval'."
            )
