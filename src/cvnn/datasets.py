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


class FlatCMRxRecon(Dataset):
    def __init__(self, root: str, transform=None, max_samples: Optional[int] = 10000):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []
        self.max_samples = max_samples
        
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]
        
        if len(subdirs) > 0:
            subdirs.sort()
            self.classes = [d.name for d in subdirs]
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for d in subdirs:
                label = self.class_to_idx[d.name]
                for file_path in d.glob("*.pt"):
                    self.samples.append((file_path, label))
        else:
            self.classes = ["data"]
            self.class_to_idx = {"data": 0}
            for file_path in self.root.glob("*.pt"):
                self.samples.append((file_path, 0))
        
        self.samples = self.samples[:self.max_samples]
                
        print(f"Dataset chargé : {len(self.samples)} images. Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        data = torch.load(path)
        
        if self.transform:
            data = self.transform(data)
            
        return data, label
                            
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
    
class MSTARClean(Dataset):
    """
    Clean MSTAR Dataset.
    Loads data from pickle, filters specific targets, and delegates 
    preprocessing (Resizing/Cropping) to the `transform` argument.
    """

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = pathlib.Path(root)
        self.transform = transform

        pkl_path = self.root / 'data.pkl'
        if not pkl_path.exists():
            raise FileNotFoundError(f"Could not find data.pkl at {pkl_path}")
        
        df = pd.read_pickle(pkl_path)

        all_classes = df.TargetType.unique()
        self.class_names = np.delete(all_classes, np.argwhere(all_classes == 'slicey')).tolist()
        self.class_to_idx = {cla: i for i, cla in enumerate(self.class_names)}
        
        mask = df['TargetType'] != 'slicey'
        
        df_filtered = df[mask].copy()

        mask_bmp2_bad = (df_filtered['TargetType'] == 'bmp2_tank') & \
                        (~df_filtered['path'].str.endswith('SN_C21'))
        
        mask_t72_bad = (df_filtered['TargetType'] == 't72_tank') & \
                       (~df_filtered['path'].str.endswith('SN_132'))
        
        df_final = df_filtered[~(mask_bmp2_bad | mask_t72_bad)]

        self.samples = df_final['data'].tolist()
        self.labels = df_final['TargetType'].map(self.class_to_idx).tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        data = self.samples[index]
        label = self.labels[index]

        data = data[np.newaxis, :, :] # (1, H, W)

        if self.transform is not None:
            data = self.transform(data)
        
        if data.is_complex():
            real = data.real
            imag = data.imag

            # Application du flou sur les composantes
            clean_real = F.gaussian_blur(real, kernel_size=3, sigma=0.2)
            clean_imag = F.gaussian_blur(imag, kernel_size=3, sigma=0.2)
            
            # Reconstitution du nombre complexe propre
            clean_data = torch.complex(clean_real, clean_imag)

        else:   
            clean_data = F.gaussian_blur(data, kernel_size=3, sigma=0.2)
        

        return clean_data, label
