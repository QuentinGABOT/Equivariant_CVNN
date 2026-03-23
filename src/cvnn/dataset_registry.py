# src/cvnn/dataset_registry.py
from typing import Dict, Any

DATASET_TYPE_REGISTRY = {
    "ALOSDataset": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": False,
        "ignore_index": None,
        "num_channels": 4,
    },
    "PolSFDataset": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": True,
        "ignore_index": 0,
        "num_channels": 4,
    },
    "Sethi": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": False,
        "ignore_index": None,
        "num_channels": 4,
    },
    "Bretigny": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": True,
        "ignore_index": 0,
        "num_channels": 3,
    },
    "S1SLC": {
        "type": "polsar",
        "supports_full_image_reconstruction": False,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": True,
        "ignore_index": None,
        "num_channels": 2,
    },
}

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get dataset type information from registry."""
    if dataset_name not in DATASET_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Registered datasets: {list(DATASET_TYPE_REGISTRY.keys())}"
        )
    return DATASET_TYPE_REGISTRY[dataset_name]