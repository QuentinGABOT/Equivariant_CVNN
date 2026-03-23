# src/cvnn/data.py

# Standard library imports
import pathlib
import random
import copy
from typing import List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchcvnn.datasets import ALOSDataset, Bretigny, PolSFDataset, S1SLC

# Local imports
from cvnn.transform_registry import build_transform_pipeline
from cvnn.utils import setup_logging
from cvnn.dataset_registry import get_dataset_info
from cvnn.data_splitting import get_label_based_split_indices
from cvnn.datasets import Sethi, GenericDatasetWrapper
from cvnn.data_statistics import compute_dataset_statistics

# module-level logger
logger = setup_logging(__name__)


def _parse_dataset_config(cfg: dict) -> dict:
    """Parse and extract common dataset configuration parameters."""
    dataset_name = cfg["data"]["dataset"]["name"]

    config = {
        "dataset_name": dataset_name,
        "trainpath": cfg["data"]["dataset"]["trainpath"],
        "has_labels": cfg["data"]["dataset"].get("has_labels"),
    }

    if dataset_name in ["ALOSDataset", "Sethi", "PolSFDataset", "Bretigny"] and "patch_size" in cfg["data"]["dataset"]:
        config["patch_size"] = cfg["data"]["dataset"]["patch_size"]
        config["patch_stride"] = cfg["data"]["dataset"]["patch_stride"]

    # Add crop coordinates if available
    if dataset_name in ["ALOSDataset", "Sethi"] and "crop_coordinates" in cfg["data"]["dataset"]:
        config["crop_coordinates"] = (
            (cfg["data"]["dataset"]["crop_coordinates"]["start_row"], cfg["data"]["dataset"]["crop_coordinates"]["start_col"]),
            (cfg["data"]["dataset"]["crop_coordinates"]["end_row"], cfg["data"]["dataset"]["crop_coordinates"]["end_col"]),
        )
    
    return config


def _find_volpath(
    base_path: pathlib.Path, vol_folder: str, max_up: int = 5
) -> pathlib.Path:
    """Recursively search parent directories for the given vol_folder under base_path."""
    candidate = base_path / vol_folder
    for i in range(max_up + 1):
        candidate = pathlib.Path("../" * i) / base_path / vol_folder
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find volpath after {max_up} levels: {candidate}"
    )


def _create_dataset(cfg: dict, transform: Optional[Any] = None, dataset_config: Optional[dict] = None):
    """Create a dataset instance based on configuration."""
    if dataset_config is None:
        dataset_config = _parse_dataset_config(cfg)

    dataset_name = dataset_config["dataset_name"]

    if dataset_name == "Bretigny":
        raise NotImplementedError(
            "Bretigny requires fold specification - use _create_bretigny_dataset"
        )
    elif dataset_name == "MNIST":
        raise NotImplementedError("MNIST requires 'train' parameter - use directly torchvision.datasets.MNIST"
        )
    elif dataset_name == "CIFAR10":
        raise NotImplementedError("CIFAR10 requires 'train' parameter - use directly torchvision.datasets.CIFAR10"
        )
    elif dataset_name == "Sethi":
        return Sethi(
            root=dataset_config["trainpath"],
            transform=transform,
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
            crop_coordinates=dataset_config.get("crop_coordinates"),
        )
    elif dataset_name == "ALOSDataset":
        base_path = pathlib.Path(dataset_config["trainpath"])
        vol_folder = "VOL-ALOS2044980750-150324-HBQR1.1__A"
        trainpath = _find_volpath(base_path, vol_folder)

        return ALOSDataset(
            volpath=trainpath,
            transform=transform,
            crop_coordinates=dataset_config.get("crop_coordinates"),
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
        )
    elif dataset_name == "PolSFDataset":
        return PolSFDataset(
            root=dataset_config["trainpath"],
            transform=transform,
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
        )
    elif dataset_name == "S1SLC":
        return S1SLC(
            root=dataset_config["trainpath"], transform=transform, lazy_loading=False
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def _create_bretigny_dataset(
    cfg: dict, fold: str, transform: Optional[Any] = None, dataset_config: Optional[dict] = None
) -> Any:
    """Create a Bretigny dataset instance for a specific fold."""
    if dataset_config is None:
        dataset_config = _parse_dataset_config(cfg)

    return Bretigny(
        root=dataset_config["trainpath"],
        fold=fold,
        transform=transform,
        patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
        patch_stride=(dataset_config["patch_stride"], dataset_config["patch_stride"]),
        keep_labels=True if cfg["task"] == "segmentation" else False,
    )


def validate_and_correct_config(cfg: dict) -> dict:
    """Validate and auto-correct configuration based on dataset type."""
    if "data" not in cfg or "dataset" not in cfg["data"]:
        return cfg
    
    transforms = cfg["data"].get("transforms", [])
    transform_names = [t["name"].lower() if "name" in t else "" for t in transforms]
    
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
        
    if "fft" in transform_names or "fft2" in transform_names:
        cfg["evaluation"]["domain"] = "spectral"
        logger.info("Auto-configured evaluation domain to 'spectral' due to FFT transform.")
    else:
        if "domain" not in cfg["evaluation"]:
            cfg["evaluation"]["domain"] = "spatial"

    if "model" not in cfg or "layer_mode" not in cfg["model"]:
        return cfg

    dataset_name = cfg["data"]["dataset"]["name"]
    dataset_info = get_dataset_info(dataset_name)

    # 2. Configurer les métadonnées structurelles
    cfg["data"]["has_labels"] = dataset_info["has_labels"]
    cfg["data"]["supports_full_image_reconstruction"] = dataset_info["supports_full_image_reconstruction"]
    cfg["data"]["type"] = dataset_info["type"]

    cfg["data"]["num_channels"] = dataset_info["num_channels"]
    
    if dataset_info.get("ignore_index") is not None:
        cfg["data"]["ignore_index"] = dataset_info["ignore_index"]

    # Validate layer_mode and pipelines
    layer_mode = cfg["model"]["layer_mode"]
    if layer_mode not in dataset_info["valid_layer_modes"]:
        default_mode = dataset_info["valid_layer_modes"][0]
        logger.warning(f"Invalid layer_mode '{layer_mode}'. Using '{default_mode}'.")
        cfg["model"]["layer_mode"] = default_mode
        layer_mode = default_mode

    real_pipeline_type = cfg["data"].get("real_pipeline_type")
    if layer_mode == "real":
        if real_pipeline_type is None:
            default_pipeline = dataset_info["default_real_pipeline"]
            cfg["data"]["real_pipeline_type"] = default_pipeline
        elif real_pipeline_type in dataset_info["invalid_real_pipelines"]:
            default_pipeline = dataset_info["default_real_pipeline"]
            logger.warning(f"Invalid pipeline '{real_pipeline_type}'. Using '{default_pipeline}'.")
            cfg["data"]["real_pipeline_type"] = default_pipeline

    return cfg


def infer_channels_from_dataloader(dataloader: DataLoader) -> int:
    """Infer number of channels by sampling from the dataloader."""
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, dict):
        sample_data = sample_batch.get("data", list(sample_batch.values())[0])
    elif isinstance(sample_batch, (list, tuple)):
        sample_data = sample_batch[0]
    else:
        sample_data = sample_batch

    if len(sample_data.shape) == 4:
        return sample_data.shape[1]
    elif len(sample_data.shape) == 3:
        return sample_data.shape[0]
    else:
        raise ValueError(f"Unexpected data shape: {sample_data.shape}")


def infer_input_size_from_dataloader(dataloader: DataLoader) -> int:
    """Infer input size from the dataloader by inspecting a batch."""
    for sample_batch in dataloader:
        if isinstance(sample_batch, torch.Tensor):
            inputs = sample_batch
        elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            inputs = sample_batch[0]
        elif isinstance(sample_batch, dict) and "inputs" in sample_batch:
            inputs = sample_batch["inputs"]
        else:
            raise ValueError(f"Expected input/target pair, got: {type(sample_batch)}")

        if isinstance(inputs, torch.Tensor):
            return inputs.shape[-2]
        else:
            raise ValueError("Could not infer input size from dataloader")


def infer_classes_from_dataloader(dataloader: DataLoader) -> int:
    """Infer number of classes by sampling labels."""
    unique_labels = set()
    max_samples = min(10, len(dataloader))

    for i, sample_batch in enumerate(dataloader):
        if i >= max_samples:
            break
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            targets = sample_batch[1]
        elif isinstance(sample_batch, dict) and "targets" in sample_batch:
            targets = sample_batch["targets"]
        elif isinstance(sample_batch, dict) and "labels" in sample_batch:
            targets = sample_batch["labels"]
        else:
            raise ValueError(f"Expected targets/labels, got: {type(sample_batch)}")

        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
        unique_labels.update(np.unique(targets_np).tolist())

    if not unique_labels:
        raise ValueError("No labels found in the dataset")

    return len(unique_labels)

def get_dataloaders(cfg: dict, use_cuda: bool) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders based on configuration."""
    dataset_config = _parse_dataset_config(cfg)
    task = cfg.get("task")

    def _get_safe_config(original_cfg):
        cfg_safe = copy.deepcopy(original_cfg)
        if "transforms" in cfg_safe["data"]:
            input_transforms = cfg_safe["data"]["transforms"]
            # Remove transforms that require stats (Normalize)
            cfg_safe["data"]["transforms"] = [
                t for t in input_transforms 
                if t["name"].lower() not in ["normalize", "logamplitude", "global_scalar_normalize"]
            ]
        return cfg_safe

    cfg_split = _get_safe_config(cfg)
    temp_transform = build_transform_pipeline(cfg_split, for_stats=True)
    
    if dataset_config["dataset_name"] == "Bretigny":
        # Bretigny manages its own splits via 'fold'
        train_indices = None 
        valid_indices = None
        test_indices = None
    elif dataset_config["dataset_name"] == "MNIST":
        # MNIST has predefined train/test splits
        train_indices = None
        valid_indices = None
        test_indices = None
    elif dataset_config["dataset_name"] == "CIFAR10":
        # CIFAR10 has predefined train/test splits
        train_indices = None
        valid_indices = None
        test_indices = None
    else:
        # For others, instantiate to calculate the split
        base_dataset_for_split = _create_dataset(cfg, temp_transform, dataset_config)
        
        if dataset_config["has_labels"] and task in ["segmentation", "classification"]:
            logger.info("Using label-based clustering split")
            train_indices, valid_indices, test_indices = get_label_based_split_indices(
                base_dataset_for_split, task, cfg
            )
        else:
            logger.info("Using random split")
            indices = list(range(len(base_dataset_for_split)))
            random.shuffle(indices)
            test_ratio = cfg["data"].get("test_ratio")
            num_valid = int(cfg["data"]["valid_ratio"] * len(indices))
            num_test = int(test_ratio * len(indices))
            
            num_train = len(indices) - num_valid - num_test
            train_indices = indices[:num_train]
            valid_indices = indices[num_train : num_train + num_valid]
            test_indices = indices[num_train + num_valid :] if test_ratio > 0 else None

    if cfg["data"].get("recompute_statistics"):
        logger.info("Recomputing statistics requested...")
        
        cfg_stats = _get_safe_config(cfg)
        stat_transform = build_transform_pipeline(cfg_stats, for_stats=True)
        
        if dataset_config["dataset_name"] == "Bretigny":
            stat_dataset = _create_bretigny_dataset(cfg, "train", stat_transform, dataset_config)
        else:
            base_dataset_stat = _create_dataset(cfg, stat_transform, dataset_config)
            stat_dataset = Subset(base_dataset_stat, train_indices)
        
            
        stats = compute_dataset_statistics(stat_dataset, num_workers=cfg["data"]["num_workers"])

        cfg["data"].update(stats)
        cfg["data"]["recompute_statistics"] = False
        
        logger.info(f"Updated config with recomputed stats: {stats}")

    train_transform = build_transform_pipeline(cfg, is_train=True)
    eval_transform = build_transform_pipeline(cfg, is_train=False)

    logger.info(f"Creating datasets with transforms: {[train_transform, eval_transform]}")

    if dataset_config["dataset_name"] == "Bretigny":
        train_dataset = _create_bretigny_dataset(cfg, "train", train_transform, dataset_config)
        valid_dataset = _create_bretigny_dataset(cfg, "valid", eval_transform, dataset_config)
    else:
        base_train_dataset = _create_dataset(cfg, train_transform, dataset_config)
        base_eval_dataset  = _create_dataset(cfg, eval_transform, dataset_config)
        
        train_dataset = Subset(base_train_dataset, train_indices)
        valid_dataset = Subset(base_eval_dataset, valid_indices)
        if test_indices is not None:
            test_dataset = Subset(base_eval_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=cfg["data"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )
    
    cfg["data"]["inferred_input_size"] = infer_input_size_from_dataloader(train_loader)
    cfg["data"]["inferred_input_channels"] = infer_channels_from_dataloader(train_loader)

    if cfg["data"].get("has_labels"):
        inferred_classes = infer_classes_from_dataloader(train_loader)
        cfg["model"]["inferred_num_classes"] = inferred_classes
        logger.info(f"Inferred {inferred_classes} classes from data")
    
    if test_indices is not None or (dataset_config["dataset_name"] in ["Bretigny", "MNIST", "CIFAR10"]):
        if dataset_config["dataset_name"] == "Bretigny":
            test_dataset = _create_bretigny_dataset(cfg, "test", eval_transform, dataset_config)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=use_cuda,
        )
        return train_loader, valid_loader, test_loader

    return train_loader, valid_loader


def get_full_image_dataloader(cfg: dict, use_cuda: bool) -> Tuple[DataLoader, int, int]:
    """Get a DataLoader for the full image dataset."""
    
    input_transform = build_transform_pipeline(cfg)

    dataset_config = _parse_dataset_config(cfg)
    nsamples_per_cols = 0
    nsamples_per_rows = 0

    if dataset_config["dataset_name"] == "Bretigny":
        base_dataset = _create_bretigny_dataset(
            cfg, "all", input_transform, dataset_config
        )
        nsamples_per_cols = base_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.nsamples_per_rows

    elif dataset_config["dataset_name"] in ["ALOSDataset", "PolSFDataset"]:
        if "crop_coordinates" not in dataset_config:
            dataset_config["crop_coordinates"] = ((0, 0), (9000, 5000))
            
        base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        
        if dataset_config["dataset_name"] == "ALOSDataset":
            nsamples_per_cols = base_dataset.nsamples_per_cols
            nsamples_per_rows = base_dataset.nsamples_per_rows
        elif dataset_config["dataset_name"] == "PolSFDataset":
            nsamples_per_cols = base_dataset.alos_dataset.nsamples_per_cols
            nsamples_per_rows = base_dataset.alos_dataset.nsamples_per_rows
            
    elif dataset_config["dataset_name"] == "Sethi":
        if "crop_coordinates" not in dataset_config:
            dataset_config["crop_coordinates"] = ((0, 0), (9000, 9000))
        base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        nsamples_per_cols = base_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.nsamples_per_rows

    logger.info(f"Full image data loader with {len(base_dataset)} segments")
    wrapped_dataset = GenericDatasetWrapper(base_dataset)
    
    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )
    return data_loader, nsamples_per_rows, nsamples_per_cols


def get_dataset_split_indices(
    cfg: dict,
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    Get the train/validation/test split indices used by get_dataloaders.
    Replicates the splitting logic solely for visualization purposes.
    """
    def _get_safe_config(original_cfg):
        cfg_safe = copy.deepcopy(original_cfg)
        if "transforms" in cfg_safe["data"]:
            input_transforms = cfg_safe["data"]["transforms"]
            cfg_safe["data"]["transforms"] = [
                t for t in input_transforms 
                if t["name"].lower() not in ["normalize", "complexnorm", "whitening"]
            ]
        return cfg_safe

    cfg_safe = _get_safe_config(cfg)
    input_transform = build_transform_pipeline(cfg_safe)
    dataset_config = _parse_dataset_config(cfg)

    if dataset_config["dataset_name"] == "Bretigny":
        train_dataset = _create_bretigny_dataset(cfg, "train", input_transform, dataset_config)
        valid_dataset = _create_bretigny_dataset(cfg, "valid", input_transform, dataset_config)
        test_dataset = _create_bretigny_dataset(cfg, "test", input_transform, dataset_config)

        total_train = len(train_dataset)
        total_valid = len(valid_dataset)
        total_test = len(test_dataset)

        train_indices = list(range(total_train))
        valid_indices = list(range(total_train, total_train + total_valid))
        test_indices = list(
            range(total_train + total_valid, total_train + total_valid + total_test)
        )
        return train_indices, valid_indices, test_indices

    else:
        try:
            base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        except FileNotFoundError:
            logger.warning(f"Dataset path not found. Cannot get split indices.")
            return [], [], None

        task = cfg.get("task", "")
        if dataset_config["dataset_name"] in ["PolSFDataset", "S1SLC"] and task in ["segmentation", "classification"]:
            return get_label_based_split_indices(base_dataset, task, cfg)
        else:
            indices = list(range(len(base_dataset)))
            random.shuffle(indices)
            test_ratio = cfg["data"].get("test_ratio")
            num_valid = int(cfg["data"]["valid_ratio"] * len(indices))
            num_test = int(test_ratio * len(indices))
            num_train = len(indices) - num_valid - num_test
            
            train_indices = indices[:num_train]
            valid_indices = indices[num_train : num_train + num_valid]
            test_indices = indices[num_train + num_valid :] if test_ratio > 0 else None

            return train_indices, valid_indices, test_indices