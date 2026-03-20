"""
cvnn.plugins

Plugin registry for experiment subclasses.
"""

from typing import Dict, Type, Any

# Registry mapping task names to Experiment classes
_PLUGINS: Dict[str, Type[Any]] = {}


def register_plugin(task: str):
    """Decorator to register an Experiment subclass for a given task."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        _PLUGINS[task] = cls
        return cls

    return decorator


def get_plugins() -> Dict[str, Type[Any]]:
    """Return all registered plugins."""
    return dict(_PLUGINS)
