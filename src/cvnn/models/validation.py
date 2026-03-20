"""Model configuration validation utilities."""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional


class ModelConfigValidator:
    """Validates model configuration parameters."""

    @staticmethod
    def validate_autoencoder_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AutoEncoder configuration."""
        validated = config.copy()

        # Required parameters
        required_params = [
            "num_channels",
            "num_layers",
            "channels_ratio",
            "input_size",
            "activation",
        ]
        for param in required_params:
            if param not in validated:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate types and ranges
        if validated["num_channels"] <= 0:
            raise ValueError("num_channels must be positive")

        if validated["num_layers"] <= 0:
            raise ValueError("num_layers must be positive")

        if validated["channels_ratio"] <= 0:
            raise ValueError("channels_ratio must be positive")

        if validated["input_size"] <= 0:
            raise ValueError("input_size must be positive")

        # Validate conv_mode
        conv_mode = validated.get("conv_mode", "complex")
        if conv_mode not in ["complex", "real", "split", "dual"]:
            raise ValueError(f"Invalid conv_mode: {conv_mode}")
        validated["conv_mode"] = conv_mode

        # Validate activation
        valid_activations = ["modReLU", "ReLU", "CReLU", "zReLU", "CPReLU"]
        if validated["activation"] not in valid_activations:
            warnings.warn(
                f"Activation '{validated['activation']}' may not be supported. "
                f"Supported activations: {valid_activations}"
            )

        # Check if input_size is compatible with num_layers
        min_size = 2 ** validated["num_layers"]
        if validated["input_size"] < min_size:
            warnings.warn(
                f"input_size ({validated['input_size']}) might be too small for "
                f"{validated['num_layers']} layers. Minimum recommended: {min_size}"
            )

        return validated

    @staticmethod
    def get_model_memory_estimate(config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate model memory usage (rough approximation)."""
        num_channels = config["num_channels"]
        num_layers = config["num_layers"]
        channels_ratio = config["channels_ratio"]
        input_size = config["input_size"]
        conv_mode = config.get("conv_mode", "complex")

        # Rough parameter count estimation
        total_params = 0
        current_channels = channels_ratio

        # First layer
        total_params += num_channels * current_channels * 9 * 2  # 3x3 conv, DoubleConv

        # Encoder layers
        for i in range(1, num_layers):
            out_channels = channels_ratio * (2**i)
            total_params += current_channels * out_channels * 9 * 2  # Down block
            current_channels = out_channels

        # Decoder layers
        for i in range(num_layers - 2, -1, -1):
            out_channels = channels_ratio * (2**i)
            total_params += current_channels * out_channels * 4  # TransposeConv
            total_params += out_channels * out_channels * 9 * 2  # DoubleConv
            current_channels = out_channels

        # Output layer
        total_params += current_channels * num_channels

        # Complex mode doubles parameters
        if conv_mode == "complex":
            total_params *= 2

        # Estimate memory (rough)
        param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float
        activation_memory_mb = (
            input_size * input_size * num_channels * 4 / (1024 * 1024)
        )

        return {
            "estimated_parameters": total_params,
            "parameter_memory_mb": param_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "total_memory_mb": param_memory_mb + activation_memory_mb,
        }


def validate_model_config(model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration for specific model type."""
    validator = ModelConfigValidator()

    if model_type.lower() in ["autoencoder", "ae", "latent_autoencoder", "latent_ae"]:
        return validator.validate_autoencoder_config(config)
    else:
        warnings.warn(f"No specific validation available for model type: {model_type}")
        return config
