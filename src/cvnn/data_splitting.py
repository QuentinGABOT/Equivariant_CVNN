# src/cvnn/data_splitting.py

# Standard library imports
from typing import List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Local imports
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


def calculate_class_distribution(base_dataset: Any, task: str) -> np.ndarray:
    """
    Compute per-sample class distributions for a dataset.

    For `classification` tasks, this returns a one-hot vector per sample.
    For `segmentation` tasks, this returns a normalized histogram of
    class frequencies for each sample (flattened mask).

    Args:
        base_dataset: an indexable dataset.
        task: 'classification' or 'segmentation'.

    Returns:
        np.ndarray of shape (n_samples, num_classes).
    """
    # Collect labels/masks from the dataset.
    samples = [base_dataset[i][1] for i in range(len(base_dataset))]

    if len(samples) == 0:
        return np.zeros((0, 0), dtype=float)

    # Convert tensors to numpy arrays for processing
    if isinstance(samples[0], torch.Tensor):
        samples = [s.cpu().numpy() for s in samples]

    # Infer set of classes present in the data and map them to indices
    all_labels = set()
    for s in samples:
        try:
            all_labels.update(np.unique(s))
        except Exception:
            all_labels.add(s)
    all_labels = sorted(all_labels)
    num_classes = len(all_labels)
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    logger.info(f"Found {num_classes} classes: {all_labels}")

    distributions = []
    if num_classes == 0:
        return np.zeros((0, 0), dtype=float)

    all_labels_arr = np.asarray(all_labels)

    if task == "classification":
        # Return one-hot vectors per sample
        for s in samples:
            idx = label_to_idx[int(np.asarray(s).item())]
            one_hot = np.zeros(num_classes, dtype=float)
            one_hot[idx] = 1.0
            distributions.append(one_hot)
            
    elif task == "segmentation":
        # Compute per-sample normalized histograms
        for mask in samples:
            mask_flat = np.array(mask).ravel()
            mapped = np.searchsorted(all_labels_arr, mask_flat)
            hist = np.bincount(mapped, minlength=num_classes).astype(float)
            hist /= hist.sum() if hist.sum() > 0 else 1.0
            distributions.append(hist)
    else:
        raise ValueError(f"Unsupported task for class distribution: {task}")

    return np.array(distributions)


def get_label_based_split_indices(
    base_dataset: Any, task: str, cfg: dict, random_state: int = 42
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    Get train/valid/test split indices based on label distribution using
    clustering/stratification on per-sample class distributions.

    Args:
        base_dataset: dataset instance containing labels/masks.
        task: 'segmentation' or 'classification'.
        cfg: configuration dict containing `data.valid_ratio` and optionally `data.test_ratio`.
        random_state: integer seed for reproducible shuffling.

    Returns:
        Tuple (train_indices, valid_indices, test_indices).
    """
    logger.info("Extracting labels for clustering-based split...")

    test_ratio = cfg["data"].get("test_ratio", 0.0)
    valid_ratio = cfg["data"]["valid_ratio"]
    rng = np.random.RandomState(random_state)

    if task == "classification":
        labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])
        labels = np.asarray(labels).reshape(-1)
        indices = np.arange(len(labels))

        if test_ratio > 0:
            # Split Test
            sss_test = StratifiedShuffleSplit(
                n_splits=1, test_size=test_ratio, random_state=rng.randint(2**32)
            )
            train_valid_idx, test_idx = next(sss_test.split(indices, labels))

            # Split Train/Valid
            remaining_labels = labels[train_valid_idx]
            valid_frac_remaining = (
                valid_ratio / (1.0 - test_ratio) if (1.0 - test_ratio) > 0 else 0.0
            )
            sss_valid = StratifiedShuffleSplit(
                n_splits=1,
                test_size=valid_frac_remaining,
                random_state=rng.randint(2**32),
            )
            train_idx_rel, valid_idx_rel = next(
                sss_valid.split(train_valid_idx, remaining_labels)
            )

            train_indices = train_valid_idx[train_idx_rel].tolist()
            valid_indices = train_valid_idx[valid_idx_rel].tolist()
            test_indices = test_idx.tolist()
            
            logger.info(f"Stratified classification split: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}")
            return train_indices, valid_indices, test_indices
        else:
            # Two-way split
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=valid_ratio, random_state=rng.randint(2**32)
            )
            train_idx, valid_idx = next(sss.split(indices, labels))
            train_indices = indices[train_idx].tolist()
            valid_indices = indices[valid_idx].tolist()
            
            logger.info(f"Stratified classification split (no test): train={len(train_indices)}, valid={len(valid_indices)}")
            return train_indices, valid_indices, None

    if task == "segmentation":
        histograms = calculate_class_distribution(base_dataset, "segmentation")
        indices = np.arange(histograms.shape[0])

        if test_ratio > 0:
            msss_test = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=test_ratio, random_state=rng.randint(2**32)
            )
            train_valid_idx, test_idx = next(msss_test.split(indices, histograms))

            remaining = histograms[train_valid_idx]
            valid_frac_remaining = (
                valid_ratio / (1.0 - test_ratio) if (1.0 - test_ratio) > 0 else 0.0
            )
            msss_valid = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=valid_frac_remaining,
                random_state=rng.randint(2**32),
            )
            train_idx_rel, valid_idx_rel = next(
                msss_valid.split(train_valid_idx, remaining)
            )

            train_indices = train_valid_idx[train_idx_rel].tolist()
            valid_indices = train_valid_idx[valid_idx_rel].tolist()
            test_indices = test_idx.tolist()
            
            logger.info(f"Multilabel stratified segmentation split: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}")
            return train_indices, valid_indices, test_indices
        else:
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=valid_ratio, random_state=rng.randint(2**32)
            )
            train_idx, valid_idx = next(msss.split(indices, histograms))
            train_indices = indices[train_idx].tolist()
            valid_indices = indices[valid_idx].tolist()
            
            logger.info(f"Multilabel stratified segmentation split (no test): train={len(train_indices)}, valid={len(valid_indices)}")
            return train_indices, valid_indices, None

    return [], [], None