# src/cvnn/custom_datasets.py
from cProfile import label
import pathlib
import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import Dataset
from typing import Union, Optional, Any, Tuple
import pandas as pd
from torchvision.transforms import functional as F
import random

class Sethi(Dataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        transform: Optional[Any] = None,
        patch_size: Tuple[int, int] = (128, 128),
        patch_stride: Optional[Tuple[int, int]] = None,
        crop_coordinates: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride else patch_size

        self.HH = np.load(self.root / "HH.npy")
        self.HV = np.load(self.root / "HV.npy")
        self.VH = np.load(self.root / "VH.npy")
        self.VV = np.load(self.root / "VV.npy")
        
        if crop_coordinates is not None:
            self.crop_coordinates = crop_coordinates
            # Apply crop
            r0, r1 = self.crop_coordinates[0][0], self.crop_coordinates[1][0]
            c0, c1 = self.crop_coordinates[0][1], self.crop_coordinates[1][1]
            self.HH = self.HH[r0:r1, c0:c1]
            self.HV = self.HV[r0:r1, c0:c1]
            self.VH = self.VH[r0:r1, c0:c1]
            self.VV = self.VV[r0:r1, c0:c1]
        else:
            self.crop_coordinates = ((0, 0), (self.HH.shape[0], self.HH.shape[1]))

        nrows = self.crop_coordinates[1][0] - self.crop_coordinates[0][0]
        ncols = self.crop_coordinates[1][1] - self.crop_coordinates[0][1]
        
        self.nsamples_per_rows = (nrows - self.patch_size[0]) // self.patch_stride[0] + 1
        self.nsamples_per_cols = (ncols - self.patch_size[1]) // self.patch_stride[1] + 1

    def __len__(self) -> int:
        return self.nsamples_per_rows * self.nsamples_per_cols

    def __getitem__(self, idx) -> Any:
        row_stride, col_stride = self.patch_stride
        start_row = (idx // self.nsamples_per_cols) * row_stride
        start_col = (idx % self.nsamples_per_cols) * col_stride
        h, w = self.patch_size
        
        # Stack channels
        patches = np.stack([
            comp[start_row : start_row + h, start_col : start_col + w]
            for comp in [self.HH, self.HV, self.VH, self.VV]
        ])

        if self.transform is not None:
            patches = self.transform(patches)

        return patches
                            
class GenericDatasetWrapper(Dataset):
    """Wrapper to handle (data, target) vs (data) outputs consistently."""
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        if isinstance(data, tuple):
            return data[0], data[1], index
        return data, index

    def __len__(self) -> int:
        return len(self.dataset)