"""Model registry for easier model management and extensibility."""

# Standard library imports
from typing import Any, Dict, Optional, Type

# Third-party imports
import torch.nn as nn

# Local imports
from .models import AutoEncoder, LatentAutoEncoder

# Global model registry
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls: Type[nn.Module]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> Type[nn.Module]:
    """Get a model class by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def create_model(name: str, **kwargs) -> nn.Module:
    """Create a model instance by name with given parameters."""
    model_cls = get_model(name)
    return model_cls(**kwargs)


# Register built-in models
register_model("AutoEncoder")(AutoEncoder)
register_model("LatentAutoEncoder")(LatentAutoEncoder)

# Add aliases for backward compatibility
_MODEL_REGISTRY["autoencoder"] = AutoEncoder
_MODEL_REGISTRY["ae"] = AutoEncoder
_MODEL_REGISTRY["latent_autoencoder"] = LatentAutoEncoder
_MODEL_REGISTRY["latent_ae"] = LatentAutoEncoder
