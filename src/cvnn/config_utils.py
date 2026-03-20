"""
Configuration utility functions for consistent config access patterns.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Optional

# Local imports
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


def get_model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common model parameters from config.

    Returns:
        Dictionary with standardized model parameters
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Handle input channels with fallback for both naming conventions
    input_channels = (
        data_cfg.get("inferred_input_channels")
        or model_cfg.get("input_channels")
        or data_cfg.get("input_channels")
    )
    if input_channels is None:
        raise ValueError(
            "No input_channels found in config (expected in data.inferred_input_channels, model.input_channels, or data.input_channels)"
        )

    params = {
        "num_channels": input_channels,
        "input_size": data_cfg.get("inferred_input_size"),
        "num_layers": model_cfg.get("num_layers"),
        "channels_width": model_cfg.get("channels_width"),
        "activation": model_cfg.get("activation"),
        "layer_mode": model_cfg.get("layer_mode"),
        "num_blocks": model_cfg.get("num_blocks"),
        "normalization_layer": model_cfg.get("normalization_layer"),
        "downsampling_layer": model_cfg.get("downsampling_layer"),
        "upsampling_layer": model_cfg.get("upsampling_layer", None),
        "residual": model_cfg.get("residual"),
        "dropout": model_cfg.get("dropout"),
        "projection_layer": model_cfg.get("projection_layer", None),
        "projection_config": model_cfg.get("projection", {}),
        "gumbel_softmax": model_cfg.get("gumbel_softmax", None),
    }

    # Add task-specific parameters
    if "inferred_num_classes" in model_cfg:
        params["num_classes"] = model_cfg["inferred_num_classes"]

    if "latent_dim" in model_cfg:
        params["latent_dim"] = model_cfg["latent_dim"]
    
    if "cov_mode" in model_cfg:
        params["cov_mode"] = model_cfg["cov_mode"]
    
    if "force_circular" in model_cfg:
        params["force_circular"] = model_cfg["force_circular"]

    if "standard_reparam" in model_cfg:
        params["standard_reparam"] = model_cfg["standard_reparam"]
    
    if "use_conv_1x1" in model_cfg:
        params["use_conv_1x1"] = model_cfg["use_conv_1x1"]
    
    if "decoder_variance" in model_cfg:
        if "learned_variance" in model_cfg["decoder_variance"]:
            params["learned_variance"] = model_cfg["decoder_variance"]["learned_variance"]

    if "sample_gmm" in model_cfg:
        params["sample_gmm"] = model_cfg["sample_gmm"]
        
    return params


def get_wandb_config(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract WandB configuration if present."""
    logging_cfg = cfg.get("logging", {})
    if "wandb" not in logging_cfg:
        return None
    return logging_cfg["wandb"]


def get_logdir(cfg: Dict[str, Any]) -> Optional[Path]:
    """Extract log directory from config."""
    logging_cfg = cfg.get("logging", {})
    if "logdir" in logging_cfg:
        return Path(logging_cfg["logdir"])
    return None


def validate_required_config_sections(cfg: Dict[str, Any]) -> None:
    """Validate that required config sections are present."""
    required_sections = ["data", "model"]

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    # Validate required data fields
    data_required = ["dataset"]
    for field in data_required:
        if field not in cfg["data"]:
            raise ValueError(f"Missing required data config field: {field}")

    # Validate dataset has name
    if "name" not in cfg["data"]["dataset"]:
        raise ValueError("Missing required dataset name in data.dataset.name")


def update_config_with_inferred_values(cfg: Dict[str, Any], **kwargs) -> None:
    """
    Update config with inferred values from data pipeline.

    Args:
        cfg: Configuration dictionary to update
        **kwargs: Key-value pairs to set in appropriate config sections
    """
    # Map of parameter names to their config locations
    location_map = {
        "inferred_input_channels": ("data", "inferred_input_channels"),
        "inferred_num_classes": ("model", "inferred_num_classes"),
    }

    for key, value in kwargs.items():
        if key in location_map:
            section, param = location_map[key]
            if section not in cfg:
                cfg[section] = {}
            cfg[section][param] = value
            logger.debug(f"Updated config: {section}.{param} = {value}")
        else:
            logger.warning(f"Unknown config parameter for update: {key}")
