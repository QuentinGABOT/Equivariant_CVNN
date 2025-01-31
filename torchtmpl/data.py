import numpy as np
import logging
import random
import pathlib
from os import path, listdir
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torchcvnn.datasets import ALOSDataset, PolSFDataset, Bretigny
from .datasets import Sethi, MSTAR, S1SLC, MNIST, GenericDatasetWrapper
from . import transforms as transforms_module
import torchvision

# Constants for ignore index
IGNORE_INDEX = -100


def get_transform_instance(transform_name, name_dataset, size):
    # Split the transform_name string on commas and strip whitespace
    transform_names = [name.strip() for name in transform_name.split(",")]

    # Get the classes from the module based on the transform names
    transform_instances = []

    if name_dataset == "MSTAR":
        transform_instances.append(transforms_module.ComplexResizeTransform(size))
    elif name_dataset in ["PolSFDataset", "Bretigny", "S1SLC", "ALOSDataset"]:
        transform_instances.append(transforms_module.PolSARtoArrayTransform())

    for name in transform_names:
        try:
            TransformClass = getattr(transforms_module, name)
        except AttributeError:
            TransformClass = getattr(torchvision.transforms, name)
        transform_instances.append(TransformClass())

    transform_instances.append(transforms_module.CreateTensor())

    # If there's more than one transform, compose them
    if len(transform_instances) > 1:
        return torchvision.transforms.Compose(transform_instances)
    else:
        return transform_instances[0]


def calculate_class_distribution(masks: list, num_classes: int) -> np.ndarray:
    distributions = [
        np.histogram(mask, bins=np.arange(num_classes + 1))[0] / mask.size
        for mask in masks
    ]
    return np.array(distributions)


def stratify_masks(
    masks: list, num_classes: int, n_clusters: int = 10, random_state: int = 0
) -> np.ndarray:
    distributions = calculate_class_distribution(masks, num_classes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(distributions)
    return kmeans.labels_


def get_dataloaders(data_config: dict, use_cuda: bool, dtype) -> tuple:
    (
        img_size,
        img_stride,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    ) = extract_data_config(data_config)
    assert valid_ratio + test_ratio < 1.0

    class_weights, num_channels, num_classes, ignore_index = (
        None,
        None,
        None,
        IGNORE_INDEX,
    )

    if name_dataset in ["Bretigny", "PolSFDataset"]:
        ignore_index = 0  # Set ignore index to 0 for these datasets

    logging.info("  - Dataset creation")

    input_transform = get_transform_instance(transform, name_dataset, img_size)

    if name_dataset == "ALOSDataset":
        train_dataset, valid_dataset, test_dataset, _, _, _, _ = prepare_alos_dataset(
            data_config,
            img_size,
            img_stride,
            trainpath,
            input_transform,
            valid_ratio,
            test_ratio,
        )
    elif name_dataset == "Bretigny":
        train_dataset, valid_dataset, test_dataset = prepare_bretigny_dataset(
            trainpath, img_size, img_stride, input_transform, data_config
        )
    elif name_dataset == "MSTAR":
        train_dataset, valid_dataset, test_dataset = prepare_mstar_dataset(
            trainpath, img_size, input_transform, valid_ratio, test_ratio, data_config
        )
    elif name_dataset == "S1SLC":
        train_dataset, valid_dataset, test_dataset = prepare_s1slc_dataset(
            trainpath, input_transform, valid_ratio, test_ratio, data_config
        )
    elif name_dataset in ["2Shapes", "3Shapes", "MNIST_Shape"]:
        train_dataset, valid_dataset, test_dataset = prepare_MNIST_dataset(
            root=trainpath,
            input_transform=input_transform,
        )
    elif name_dataset == "PolSFDataset":
        train_dataset, valid_dataset, test_dataset, _, _, _, _ = prepare_polsfdataset(
            trainpath,
            img_size,
            img_stride,
            input_transform,
            valid_ratio,
            test_ratio,
            data_config,
        )

    if name_dataset in [
        "PolSFDataset",
        "Bretigny",
        "MSTAR",
        "S1SLC",
    ]:
        if name_dataset in ["PolSFDataset", "MSTAR", "S1SLC"]:
            num_classes = len(train_dataset.dataset.classes)
        else:
            num_classes = len(train_dataset.classes)

        class_weights = compute_class_weights(train_dataset, num_classes, ignore_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_channels = get_num_channels(train_loader)

    return (
        train_loader,
        valid_loader,
        test_loader,
        class_weights,
        num_classes,
        num_channels,
        ignore_index,
    )


def get_full_image_dataloader(data_config: dict, use_cuda: bool, dtype) -> tuple:
    (
        img_size,
        _,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    ) = extract_data_config(data_config)

    nsamples_per_cols, nsamples_per_rows = None, None

    logging.info("  - Dataset creation")

    input_transform = get_transform_instance(transform, name_dataset, img_size)

    if name_dataset == "ALOSDataset":
        (
            train_dataset,
            valid_dataset,
            test_dataset,
            base_dataset,
            train_indices,
            valid_indices,
            test_indices,
        ) = prepare_alos_dataset(
            data_config,
            img_size,
            img_size,
            trainpath,
            input_transform,
            valid_ratio,
            test_ratio,
        )
        nsamples_per_cols = base_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.nsamples_per_rows

    elif name_dataset == "Bretigny":
        _, _, test_dataset = prepare_bretigny_dataset(
            trainpath, img_size, img_size, input_transform, data_config
        )
        nsamples_per_cols = test_dataset.nsamples_per_cols
        nsamples_per_rows = test_dataset.nsamples_per_rows
    elif name_dataset == "PolSFDataset":
        (
            train_dataset,
            valid_dataset,
            test_dataset,
            base_dataset,
            train_indices,
            valid_indices,
            test_indices,
        ) = prepare_polsfdataset(
            trainpath,
            img_size,
            img_size,
            input_transform,
            valid_ratio,
            test_ratio,
            data_config,
        )
        nsamples_per_cols = base_dataset.alos_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.alos_dataset.nsamples_per_rows

    wrapped_dataset = GenericDatasetWrapper(base_dataset)

    data_loader = DataLoader(
        wrapped_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    if name_dataset in ["PolSFDataset", "ALOSDataset"]:
        indices = [train_indices, valid_indices, test_indices]
    else:
        indices = None

    return (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
        indices,
    )


def extract_data_config(data_config: dict) -> tuple:
    """
    Extracts necessary fields from the data configuration dictionary.
    """
    img_size = (data_config["img_size"], data_config["img_size"])
    img_stride = (data_config["img_stride"], data_config["img_stride"])
    valid_ratio = data_config["valid_ratio"]
    test_ratio = data_config["test_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    name_dataset = data_config["dataset"]["name"]
    transform = data_config["transform"]
    trainpath = path.expandvars(
        data_config["dataset"]["trainpath"]
    )  # apply the substitution if an environment variable is used
    return (
        img_size,
        img_stride,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    )


def prepare_alos_dataset(
    data_config,
    img_size,
    img_stride,
    trainpath,
    input_transform,
    valid_ratio,
    test_ratio,
):
    if "crop" in data_config.keys():
        crop_coordinates = (
            (data_config["crop"]["start_row"], data_config["crop"]["start_col"]),
            (data_config["crop"]["end_row"], data_config["crop"]["end_col"]),
        )
    else:
        crop_coordinates = None

    trainpath = pathlib.Path(trainpath) / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    base_dataset = eval(
        f"{data_config['dataset']['name']}(volpath=trainpath, transform=input_transform, crop_coordinates=crop_coordinates, patch_size=img_size, patch_stride=img_stride)"
    )
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)

    num_valid = int(valid_ratio * len(indices))
    num_test = int(test_ratio * len(indices))
    num_train = len(indices) - num_valid - num_test

    train_indices = indices[:num_train]
    valid_indices = indices[num_train : num_train + num_valid]
    test_indices = indices[num_train + num_valid :]

    train_dataset = Subset(base_dataset, train_indices)
    valid_dataset = Subset(base_dataset, valid_indices)
    test_dataset = Subset(base_dataset, test_indices)

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        base_dataset,
        train_indices,
        valid_indices,
        test_indices,
    )


def prepare_bretigny_dataset(
    trainpath, img_size, img_stride, input_transform, data_config
):
    train_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, fold='train', transform=input_transform, patch_size=img_size, patch_stride=img_stride)"
    )

    valid_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, fold='valid', transform=input_transform, patch_size=img_size, patch_stride=img_stride)"
    )

    test_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, fold='test', transform=input_transform, patch_size=img_size, patch_stride=img_stride)"
    )

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return train_dataset, valid_dataset, test_dataset


def prepare_mstar_dataset(
    trainpath, img_size, input_transform, valid_ratio, test_ratio, data_config
):
    base_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, transform=input_transform, img_size=img_size)"
    )
    labels = [base_dataset[idx][1] for idx in range(len(base_dataset))]
    train_indices, temp_indices = train_test_split(
        list(range(len(base_dataset))),
        stratify=labels,
        test_size=(valid_ratio + test_ratio),
    )

    temp_masks = [labels[i] for i in temp_indices]
    valid_indices, test_indices = train_test_split(
        temp_indices,
        stratify=temp_masks,
        test_size=(test_ratio / (valid_ratio + test_ratio)),
    )

    train_dataset = Subset(base_dataset, train_indices)
    valid_dataset = Subset(base_dataset, valid_indices)
    test_dataset = Subset(base_dataset, test_indices)

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return train_dataset, valid_dataset, test_dataset


def prepare_s1slc_dataset(
    trainpath, input_transform, valid_ratio, test_ratio, data_config
):
    base_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, transform=input_transform)"
    )
    labels = [base_dataset[idx][1] for idx in range(len(base_dataset))]
    train_indices, temp_indices = train_test_split(
        list(range(len(base_dataset))),
        stratify=labels,
        test_size=(valid_ratio + test_ratio),
    )

    temp_masks = [labels[i] for i in temp_indices]
    valid_indices, test_indices = train_test_split(
        temp_indices,
        stratify=temp_masks,
        test_size=(test_ratio / (valid_ratio + test_ratio)),
    )

    train_dataset = Subset(base_dataset, train_indices)
    valid_dataset = Subset(base_dataset, valid_indices)
    test_dataset = Subset(base_dataset, test_indices)

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return train_dataset, valid_dataset, test_dataset


def prepare_polsfdataset(
    trainpath,
    img_size,
    img_stride,
    input_transform,
    valid_ratio,
    test_ratio,
    data_config,
):
    base_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, transform=input_transform, patch_size=img_size, patch_stride=img_stride)"
    )
    num_classes = len(base_dataset.classes)
    masks = [base_dataset[i][1] for i in range(len(base_dataset))]

    strat_labels = stratify_masks(masks, num_classes)
    train_indices, temp_indices = train_test_split(
        list(range(len(base_dataset))),
        stratify=strat_labels,
        test_size=(valid_ratio + test_ratio),
    )

    temp_masks = [strat_labels[i] for i in temp_indices]
    valid_indices, test_indices = train_test_split(
        temp_indices,
        stratify=temp_masks,
        test_size=(test_ratio / (valid_ratio + test_ratio)),
    )

    train_dataset = Subset(base_dataset, train_indices)
    valid_dataset = Subset(base_dataset, valid_indices)
    test_dataset = Subset(base_dataset, test_indices)

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        base_dataset,
        train_indices,
        valid_indices,
        test_indices,
    )


def prepare_MNIST_dataset(root, input_transform):
    train_dataset = MNIST(root=root, fold="train", transform=input_transform)
    valid_dataset = MNIST(root=root, fold="val", transform=input_transform)
    test_dataset = MNIST(root=root, fold="test", transform=input_transform)
    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return train_dataset, valid_dataset, test_dataset


def compute_class_weights(train_dataset, num_classes, ignore_index):
    if isinstance(train_dataset[0][1], int):
        all_labels = torch.tensor(
            [train_dataset[idx][1] for idx in range(len(train_dataset))]
        )
    elif isinstance(train_dataset[0][1], (torch.Tensor, np.ndarray)):
        all_labels = torch.cat(
            [
                torch.from_numpy(train_dataset[idx][1].flatten())
                for idx in range(len(train_dataset))
            ]
        )
    class_counts = torch.bincount(all_labels)

    if ignore_index > 0:
        # Exclude the count of the unlabeled class if necessary (assuming class 0 is unlabeled)
        class_counts = class_counts[1:] if num_classes > 1 else class_counts

    total_count = class_counts.sum()

    # Compute class weights
    class_weights = (
        (total_count / (num_classes - (1 if ignore_index > 0 else 0)) / class_counts)
        if num_classes > 1
        else np.array([1.0])
    )

    if ignore_index > 0:
        class_weights = np.concatenate(
            (np.array([0.0]), class_weights), dtype=np.float32
        )
        class_weights = torch.from_numpy(class_weights).type(torch.float32)

    return class_weights


def get_num_channels(loader):
    """
    Get the number of channels from the first image in the dataset.
    """
    return loader.dataset[0][0].shape[0]


def get_prostate_t2_dataset():
    import argparse
    import h5py
    import numpy as np
    from pathlib import Path
    from matplotlib import pyplot as plt
    import xml.etree.ElementTree as etree
    from .fastmri_prostate.reconstruction.t2.prostate_t2_recon import t2_reconstruction
    from .fastmri_prostate.data.mri_data import load_file_T2, save_recon
    import warnings
    import os

    file_name = "/gpfs/workdir/gabotqu/datasets/fastMRI_PROSTATE_T2/fastMRI_prostate_T2_IDS_001_020/file_prostate_AXT2_001.h5"

    kspace, calibration_data, ismrmrd_header, reconstruction_rss, image_atts = (
        load_file_T2(file_name)
    )

    print("Sizes of the array fields in the fastMRI prostate T2 file:")
    print(
        "kspace:",
        kspace.shape,
        ", calibration_data:",
        calibration_data.shape,
        ", reconstruction_rss:",
        reconstruction_rss.shape,
    )

    img_dict_t2 = t2_reconstruction(kspace, calibration_data, ismrmrd_header)
    print("Size of reconstructed data:")
    print(img_dict_t2["reconstruction_rss"].shape)
    print("Type of reconstructed data:")
    print(img_dict_t2["reconstruction_rss"].dtype)

    def display_t2_slice(img, slice_num):
        plt.imshow(np.abs(img[slice_num, :, :]), cmap="gray")
        plt.title("Reconstructed T2 image - slice 11")
        plt.axis("off")
        plt.savefig(
            "/gpfs/workdir/gabotqu/complex-valued-generarive-ai-for-sar-imaging/test.png"
        )

    display_t2_slice(img_dict_t2["reconstruction_rss"], 10)

    input()

    """
    # start here
    hf = h5py.File(file_name)
    print('Keys:', list(hf.keys()))
    print('Attrs:', dict(hf.attrs))

    kspace = hf['kspace'][()]
    calibration_data = hf['calibration_data'][()]
    reconstruction_rss = hf['reconstruction_rss'][()]
    ismrmrd_header = hf['ismrmrd_header'][()]

    print("Sizes of the array fields in the fastMRI prostate T2 file:")
    print("kspace:", kspace.shape, ", calibration_data:", calibration_data.shape, ", reconstruction_rss:", reconstruction_rss.shape)

    input()
    # End here

    """


def reassemble_image(
    segments,
    samples_per_col,
    samples_per_row,
    num_channels,
    segment_size,
    real_indices,
    sets_indices=None,
):
    """
    Reassemble an image from its segments using real_indices to determine their positions.

    Args:
        segments: List or array of image segments.
        samples_per_col: Number of segments per column in the reassembled image.
        samples_per_row: Number of segments per row in the reassembled image.
        num_channels: Number of channels in the image.
        segment_size: Height/width of each square segment.
        real_indices: List of real indices corresponding to the segments.
        sets_indices: List of sets of indices for mask assignment (optional).

    Returns:
        reassembled_image: The reconstructed image tensor.
        mask: A mask indicating the set each segment belongs to (if sets_indices is provided).
    """
    # Calculate total image dimensions
    img_height = samples_per_row * segment_size
    img_width = samples_per_col * segment_size

    # Initialize the empty image tensor with the correct shape
    reassembled_image = np.zeros(
        (num_channels, img_height, img_width), dtype=segments[0].dtype
    )
    if sets_indices is None:
        mask = None
    else:
        mask = np.zeros_like(reassembled_image, dtype=np.uint8)

    # Map real_indices to their positions
    index_to_position = {
        real_index: (row, col)
        for row in range(samples_per_row)
        for col in range(samples_per_col)
        for real_index in [row * samples_per_col + col]
    }

    # Place each segment into the correct position
    for segment_index, real_index in enumerate(real_indices):
        if real_index not in index_to_position:
            raise ValueError(
                f"Real index {real_index} is out of bounds for the image grid."
            )

        # Get the target row and column
        row, col = index_to_position[real_index]
        h_start = row * segment_size
        w_start = col * segment_size

        # Insert the segment into the image
        reassembled_image[
            :, h_start : h_start + segment_size, w_start : w_start + segment_size
        ] = segments[segment_index]

        # Update the mask if sets_indices is provided
        if mask is not None:
            if real_index in sets_indices[0]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 0
            elif real_index in sets_indices[1]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 1
            elif real_index in sets_indices[2]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 2

    return reassembled_image, mask


def delete_folders_with_few_pngs(log_path, min_png_count=20):
    """
    Deletes folders under `root_path` containing fewer than `min_png_count` .png files.

    :param root_path: Path to the directory to search through.
    :param min_png_count: Minimum number of .png files a folder must contain to be kept.
    """

    for folder_name in listdir(log_path):
        folder_path = path.join(log_path, folder_name)
        if path.isdir(folder_path):  # Check if it's a directory
            png_files = [file for file in listdir(folder_path) if file.endswith(".png")]
            if len(png_files) < min_png_count:
                print(
                    f"Deleting folder: {folder_path} (contains {len(png_files)} .png files)"
                )
                shutil.rmtree(folder_path)
