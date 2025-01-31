# MIT License

# Copyright (c) 2024 Chengfang Ren, Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
import pathlib
from typing import Tuple, Any
import os

# External imports
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import tifffile as tiff


class MSTAR(Dataset):
    def __init__(self, root, img_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders of .npy images.
            transform (callable, optional): Optional transform to be applied
                on a sample.


        The dataset is composed of images of various sizes: (173, 172), (158, 158), (54, 54), (139, 138), (178, 177), (128, 128), (129, 128), (193, 192)
        As such, the images are resized to (54, 54)
        """
        self.root_dir = root
        self.transform = transform
        self.img_size = img_size
        self.data = []
        self.labels = []
        self.label_map = self.create_label_map()

        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(label_dir, file_name)
                        self.data.append(np.load(file_path))
                        self.labels.append(self.label_map[label])

        self.classes = list(set(self.labels))

    def create_label_map(self):
        """
        Creates a mapping from string labels to integer labels.
        """
        labels = sorted(os.listdir(self.root_dir))
        label_map = {label: idx for idx, label in enumerate(labels)}
        return label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class Sethi(Dataset):
    r"""
    Sethi Dataset

    Arguments:
        root: the root directory containing the npz files for Bretigny
        transform : the transform applied the cropped image
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size
    """

    def __init__(
        self,
        root: str,
        transform=None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        self.root = pathlib.Path(root)
        self.transform = transform

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.patch_stride = patch_size

        # Preload the data
        if not self.root.exists():
            raise RuntimeError(f"Cannot find the directory {self.root}")

        self.HH = np.load(self.root / "HH.npy").T[:, 0:40500]
        self.HV = np.load(self.root / "HV.npy").T[:, 0:40500]
        self.VH = np.load(self.root / "VH.npy").T[:, 0:40500]
        self.VV = np.load(self.root / "VV.npy").T[:, 0:40500]

        # Precompute the dimension of the grid of patches
        self.nrows = self.HH.shape[0]
        self.ncols = self.HH.shape[1]

        nrows_patch, ncols_patch = self.patch_size
        row_stride, col_stride = self.patch_stride

        self.nsamples_per_rows = (self.nrows - nrows_patch) // row_stride + 1
        self.nsamples_per_cols = (self.ncols - ncols_patch) // col_stride + 1

    def __len__(self) -> int:
        """
        Returns the total number of patches in the whole image.

        Returns:
            the total number of patches in the dataset
        """
        return self.nsamples_per_rows * self.nsamples_per_cols

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Returns the indexes patch.

        Arguments:
            idx (int): Index

        Returns:
            patch: where patch contains the 4 complex valued polarization HH, VH, HV, VV
        """
        row_stride, col_stride = self.patch_stride
        start_row = (idx // self.nsamples_per_cols) * row_stride
        start_col = (idx % self.nsamples_per_cols) * col_stride
        num_rows, num_cols = self.patch_size
        patches = [
            patch[
                start_row : (start_row + num_rows), start_col : (start_col + num_cols)
            ]
            for patch in [self.HH, self.HV, self.VH, self.VV]
        ]
        patches = np.stack(patches)
        if self.transform is not None:
            patches = self.transform(patches)

        return patches


class AIR_PolSAR(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.image_paths = []
        self.gt_paths = []

        # Assuming the directory structure:
        # root_dir/train/HH, HV, VH, VV, GT
        polarizations = ["HH", "HV", "VH", "VV"]
        for filename in os.listdir(os.path.join(self.root_dir, "HH")):
            self.image_paths.append(
                {
                    "HH": os.path.join(self.root_dir, "HH", filename),
                    "HV": os.path.join(self.root_dir, "HV", filename),
                    "VH": os.path.join(self.root_dir, "VH", filename),
                    "VV": os.path.join(self.root_dir, "VV", filename),
                }
            )
            self.gt_paths.append(
                os.path.join(self.root_dir, "GT", filename.replace(".tiff", ".png"))
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_dict = self.image_paths[idx]
        gt_path = self.gt_paths[idx]

        # Load the polarization images
        hh_image = tiff.imread(image_dict["HH"])
        hv_image = tiff.imread(image_dict["HV"])
        vh_image = tiff.imread(image_dict["VH"])
        vv_image = tiff.imread(image_dict["VV"])

        # Load the ground truth image
        gt_image = Image.open(gt_path).convert("L")

        # Convert the images to tensors
        hh_image = torch.from_numpy(hh_image).unsqueeze(0).float()
        hv_image = torch.from_numpy(hv_image).unsqueeze(0).float()
        vh_image = torch.from_numpy(vh_image).unsqueeze(0).float()
        vv_image = torch.from_numpy(vv_image).unsqueeze(0).float()
        gt_image = torch.from_numpy(np.array(gt_image)).long()

        # Stack the polarization images along the channel dimension
        image = torch.cat([hh_image, hv_image, vh_image, vv_image], dim=0)

        if self.transform:
            image, gt_image = self.transform((image, gt_image))

        return image, gt_image


class S1SLC(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        # Get list of subfolders in the root path
        subfolders = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]

        self.data = []
        self.labels = []

        for subfolder in subfolders:
            # Define paths to the .npy files
            hh_path = os.path.join(subfolder, "HH.npy")
            hv_path = os.path.join(subfolder, "HV.npy")
            labels_path = os.path.join(subfolder, "Labels.npy")

            # Load the .npy files
            hh = np.load(hh_path)
            hv = np.load(hv_path)
            label = np.load(labels_path)
            label = [int(l.item()) - 1 for l in label]  # Convert to 0-indexed labels

            # Concatenate HH and HV to create a two-channel array
            data = np.stack((hh, hv), axis=1)  # Shape: (B, 2, H, W)

            # Append data and labels to the lists
            self.data.extend(data)
            self.labels.extend(label)

        self.classes = list(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class MNIST(Dataset):
    def __init__(self, root, fold, transform=None):
        self.transform = transform

        if not fold in ["train", "val", "test"]:
            raise ValueError(
                f"Unrecognized fold {fold}. Should be either train, valid or test"
            )
        self.dataset = np.load(os.path.join(root, fold, "images.npy"))
        self.labels = np.load(os.path.join(root, fold, "labels.npy"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = (self.dataset[idx] + 1) / 2  # Normalize to [0, 1] range.
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class GenericDatasetWrapper(Dataset):
    def __init__(self, dataset):
        """
        A generic dataset wrapper that works with any dataset class.

        Args:
            dataset: An instance of a dataset class (e.g., CIFAR10, MNIST, etc.).
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.

        Args:
            index: Index of the item to fetch.

        Returns:
            A tuple containing (data, target, index).
        """
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)
