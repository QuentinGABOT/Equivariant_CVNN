"""
Weights & Biases utilities for consistent logging setup.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Local imports
from cvnn.config_utils import get_wandb_config
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


def setup_wandb(
    cfg: Dict[str, Any], resume_logdir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[Callable], str]:
    """
    Initialize or resume a Weights & Biases run.

    Args:
        cfg: Configuration dictionary
        resume_logdir: Optional directory for resuming runs

    Returns:
        Tuple of (wandb.log function or None, run_name)
    """
    wandb_cfg = get_wandb_config(cfg)
    if wandb_cfg is None:
        logger.info("WandB not configured, skipping initialization")
        return None, ""

    # Check if wandb is disabled via environment variable
    if os.getenv("WANDB_MODE") == "disabled":
        logger.info("WandB disabled via environment variable")
        return None, ""

    try:
        import wandb
    except ImportError:
        logger.warning("WandB not installed, skipping initialization")
        return None, ""

    # Configuration parameters
    project = wandb_cfg.get("project", "cvnn")
    mode = wandb_cfg.get("mode", "online")
    entity = wandb_cfg.get("entity")
    tags = wandb_cfg.get("tags", [])

    # Handle run resumption
    existing_id = wandb_cfg.get("run_id") if resume_logdir else None

    try:
        if existing_id:
            logger.info(f"Resuming WandB run: {existing_id}")
            run = wandb.init(
                project=project,
                entity=entity,
                config=cfg,
                id=existing_id,
                resume="must",
                mode=mode,
                tags=tags,
            )
        else:
            logger.info(f"Starting new WandB run for project: {project}")
            run = wandb.init(
                project=project, entity=entity, config=cfg, mode=mode, tags=tags
            )
            # Store run ID in config for potential resumption
            wandb_cfg["run_id"] = run.id

        run_name = run.name or wandb_cfg.get("run_id", "")
        logger.info(f"WandB initialized successfully. Run: {run_name}")
        return wandb.log, run_name

    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return None, ""


def log_model_summary(
    wandb_log: Optional[Callable], model_summary: str, step: Optional[int] = None
) -> None:
    """
    Log model summary to WandB if available.

    Args:
        wandb_log: WandB logging function (or None)
        model_summary: String representation of model
        step: Optional step number
    """
    if wandb_log is None:
        return

    try:
        log_data = {"model_summary": model_summary}
        if step is not None:
            wandb_log(log_data, step=step)
        else:
            wandb_log(log_data)
        logger.debug("Model summary logged to WandB")
    except Exception as e:
        logger.warning(f"Failed to log model summary to WandB: {e}")


def log_metrics(
    wandb_log: Optional[Callable],
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to WandB if available.

    Args:
        wandb_log: WandB logging function (or None)
        metrics: Dictionary of metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    if wandb_log is None or not metrics:
        return

    try:
        # Add prefix to metric names if provided
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics.copy()

        if step is not None:
            wandb_log(prefixed_metrics, step=step)
        else:
            wandb_log(prefixed_metrics)

        logger.debug(f"Metrics logged to WandB: {list(prefixed_metrics.keys())}")
    except Exception as e:
        logger.warning(f"Failed to log metrics to WandB: {e}")


def log_config_summary(wandb_log: Optional[Callable], cfg: Dict[str, Any]) -> None:
    """
    Log a human-readable configuration summary to WandB.

    Args:
        wandb_log: WandB logging function (or None)
        cfg: Configuration dictionary
    """
    if wandb_log is None:
        return

    try:
        # Create a formatted summary
        summary_lines = ["## Configuration Summary"]

        # Key sections to highlight
        if "data" in cfg:
            data_cfg = cfg["data"]
            summary_lines.extend(
                [
                    "### Data",
                    f"- Dataset: {data_cfg.get('dataset', {}).get('name', 'Unknown')}",
                    f"- Batch size: {data_cfg.get('batch_size', 'Unknown')}",
                    f"- Patch size: {data_cfg.get('patch_size', 'Unknown')}",
                ]
            )

        if "model" in cfg:
            model_cfg = cfg["model"]
            summary_lines.extend(
                [
                    "### Model",
                    f"- Class: {model_cfg.get('class', 'Unknown')}",
                    f"- Layers: {model_cfg.get('num_layers', 'Unknown')}",
                    f"- Mode: {model_cfg.get('layer_mode', 'Unknown')}",
                    f"- Activation: {model_cfg.get('activation', 'Unknown')}",
                ]
            )

        if "train" in cfg:
            train_cfg = cfg["train"]
            summary_lines.extend(
                [
                    "### Training",
                    f"- Epochs: {cfg.get('nepochs', 'Unknown')}",
                    f"- Learning rate: {train_cfg.get('lr', 'Unknown')}",
                    f"- Optimizer: {cfg.get('optim', {}).get('algo', 'Unknown')}",
                ]
            )

        summary_text = "\\n".join(summary_lines)
        wandb_log({"config_summary": summary_text})

        logger.debug("Configuration summary logged to WandB")
    except Exception as e:
        logger.warning(f"Failed to log config summary to WandB: {e}")


def finish_wandb_run() -> None:
    """Safely finish the WandB run if active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error finishing WandB run: {e}")
