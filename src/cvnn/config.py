"""
cvnn.config

Load and validate experiment configurations.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import yaml
from cvnn.config_utils import validate_required_config_sections
from pydantic import BaseModel, ConfigDict, model_validator

from cvnn.data import validate_and_correct_config


class ConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_real_pipeline_consistency(self):
        """Validate that real_pipeline_type requires layer_mode to be 'real'."""
        data_config = self.data
        model_config = self.model

        # Check if real_pipeline_type is specified
        real_pipeline_type = data_config.get("real_pipeline_type")
        if real_pipeline_type is not None:
            # Ensure layer_mode is 'real'
            layer_mode = model_config.get("layer_mode")
            if layer_mode != "real":
                raise ValueError(
                    f"Configuration error: When 'real_pipeline_type' is specified ('{real_pipeline_type}'), "
                    f"'layer_mode' must be set to 'real', but got '{layer_mode}'. "
                    f"Real-valued pipelines require real-valued models."
                )

            # Validate the real_pipeline_type value
            valid_pipeline_types = [
                "real_real",
                "complex_amplitude_real",
                "complex_dual_real",
            ]
            if real_pipeline_type not in valid_pipeline_types:
                raise ValueError(
                    f"Invalid 'real_pipeline_type': '{real_pipeline_type}'. "
                    f"Must be one of: {valid_pipeline_types}"
                )

        return self


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config, resolving relative paths from repository root."""

    # helper to find repository root containing .git
    def _find_repo_root(p: Path) -> Path:
        for parent in p.resolve().parents:
            if (parent / ".git").is_dir():
                return parent
        raise RuntimeError("Repository root with .git not found.")

    repo_root = _find_repo_root(Path(__file__))
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    # merge with generic config if this is a task-specific config
    generic_cfg_path = Path(repo_root) / "configs" / "config.yaml"
    if generic_cfg_path.is_file() and cfg_path.name != "config.yaml":
        with open(generic_cfg_path, "r") as gf:
            generic_cfg = yaml.safe_load(gf) or {}

        # deep merge generic and task-specific configs
        def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            for key, val in b.items():
                if key in a and isinstance(a[key], dict) and isinstance(val, dict):
                    a[key] = _deep_merge(a[key], val)
                else:
                    a[key] = val
            return a

        raw_cfg = _deep_merge(generic_cfg, raw_cfg)
    
    # validate schema and consistency
    cfg = ConfigSchema(**raw_cfg).model_dump()

    # Additional validation for config consistency
    validate_config_consistency(cfg)
    # Validate and correct based on dataset type
    cfg = validate_and_correct_config(cfg)
    # Final check for required sections
    validate_required_config_sections(cfg)

    return cfg


def validate_config_consistency(config: Dict[str, Any]) -> None:
    """Validate configuration consistency for real-valued pipelines.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If real_pipeline_type is specified but layer_mode is not 'real'

    Example:
        >>> config = {"model": {"layer_mode": "complex"}, "data": {"real_pipeline_type": "real_real"}}
        >>> validate_config_consistency(config)  # Raises ValueError

        >>> config = {"model": {"layer_mode": "real"}, "data": {"real_pipeline_type": "real_real"}}
        >>> validate_config_consistency(config)  # OK
    """
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    # Check if real_pipeline_type is specified
    real_pipeline_type = data_config.get("real_pipeline_type")
    if real_pipeline_type is not None:
        # Ensure layer_mode is 'real'
        layer_mode = model_config.get("layer_mode")
        if layer_mode != "real":
            raise ValueError(
                f"Configuration error: When 'real_pipeline_type' is specified ('{real_pipeline_type}'), "
                f"'layer_mode' must be set to 'real', but got '{layer_mode}'. "
                f"Real-valued pipelines require real-valued models."
            )

        # Validate the real_pipeline_type value
        valid_pipeline_types = [
            "real_real",
            "complex_amplitude_real",
            "complex_dual_real",
        ]
        if real_pipeline_type not in valid_pipeline_types:
            raise ValueError(
                f"Invalid 'real_pipeline_type': '{real_pipeline_type}'. "
                f"Must be one of: {valid_pipeline_types}"
            )
