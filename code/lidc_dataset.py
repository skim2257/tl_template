import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from imgtools.ops import Resample, Resize

import torchio as tio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from transforms import *
from utils import *
from math import pi

class LIDCDataset(Dataset):
    """Dataset class used in simple CNN baseline training.

    The images are loaded using SimpleITK, preprocessed and cached for faster
    retrieval during training.
    """
    def __init__(self,
                 root_directory: str,
                 input_size: tuple = (64, 64, 64),
                 split: str = "train",
                 dataaug: bool = False,
                 num_workers: int = -1,
                 resize: bool = False,
                 acsconv: bool = False,
                 mode: str = "box",
                 debug: bool = False,
                 organ: str = "lung",
                 norm: str = 'minmax'):
        """Initialize the class.

        If the cache directory does not exist, the dataset is first
        preprocessed and cached.

        Parameters
        ----------
        root_directory
            Path to directory containing the training and test images and
            segmentation masks.
        clinical_data_path
            Path to a CSV file with subject metadata and clinical information.
        input_size
            The size of input volume to extract around the tumour centre.
        train
            Whether to load the training or test set.
        dataaug
            Whether to use data augmentation.
        num_workers
            Number of parallel processes to use for data preprocessing.
        mask
            Whether to mask only contour voxels and discard background
        """
        super(LIDCDataset, self).__init__()
        self.root_directory = root_directory
        self.input_size = input_size
        self.split = split
        self.dataaug = dataaug
        self.num_workers = num_workers
        self.acsconv = acsconv
        self.mask = True
        self.resize = resize
        self.resample = Resample(spacing=(1., 1., 1.))
        self.mode = mode
        self.debug = debug
        self.norm = norm

        if organ.lower() == 'lung':
            self.low, self.high = -500, 1000
        elif organ.lower() == 'hn_soft':
            self.low, self.high = 0, 400
        elif organ.lower() == 'hn_soft_narrow':
            self.low, self.high = 50, 350
        elif organ.lower() == 'sejin_custom':
            self.low, self.high = -400, 400

        if 'nomask' in mode or mode == 'raw_img':
            self.mask = False

        # self.img_dir   = os.path.join(root_directory, 'images')
        # self.mask_dir  = os.path.join(root_directory, 'lungmasks')
        self.img_dir = os.path.join(self.root_directory, split)
        self.pat_files = os.listdir(self.img_dir)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            The input-target pair.
        """
        
        file    = self.pat_files[idx]
        img     = sitk.ReadImage(os.path.join(self.img_dir, file))

        if 'box' in self.mode:
            img     = self.resample(img)
            mask    = self.resample(mask)
        
        if self.mode == 'raw_img': # raw_img returns just the image
            img_tensor = tio.ScalarImage.from_sitk(img).data
        else:
            # bounding box or centroid
            if 'small' in self.mode:
                bbox_coords = find_bbox(mask)
            else:
                centroid = find_centroid(mask)
            
            # random translation
            if self.dataaug and 'small' not in self.mode: 
                # calculate margins
                tol = np.subtract(centroid, np.divide(self.input_size, 2))
                shape = img.GetSize()
                tol2 = np.subtract(shape, np.add(centroid, np.divide(self.input_size, 2)))
                
                # set random translation to 0 if margins too thin
                rand_translate = torch.randint(-6, 6, (3,))
                if tol[0] <= 6 or tol2[0] <= 6:
                    rand_translate[0] = 0
                if tol[1] <= 6 or tol2[1] <= 6:
                    rand_translate[1] = 0
                if tol[2] <= 6 or tol2[2] <= 6:
                    rand_translate[2] = 0

                # translate
                centroid = np.add(centroid, rand_translate)
            
            img = sitk.Clamp(img, sitk.sitkFloat32, self.low, self.high)
            
            # crop
            if 'small' in self.mode:
                img_crop = crop_bbox(img, bbox_coords, self.input_size)
                # if self.mask: mask_crop = crop_bbox(mask, bbox_coords, self.input_size)
            else:
                img_crop = crop_centroid(img, centroid, self.input_size)
                # if self.mask: mask_crop = crop_centroid(img, centroid, self.input_size)
        
            try:
                assert img_crop.GetSize() == self.input_size
            except:
                print("Image not cropped properly: ", img_crop.GetSize(), self.input_size)
        
            # if self.mask:
            #     img_crop = img_crop * sitk.Cast(mask_crop, sitk.sitkFloat32)
            
            # sitk.Image to torch.Tensor (via TorchIO)
            img_tensor = tio.ScalarImage.from_sitk(sitk.Cast(img_crop, sitk.sitkFloat32)).data

        # Q: should everything before this be int to save time?
        if self.norm == 'minmax': # min-max normalization (0 to 1)
            img_tensor = (img_tensor - self.low)/(self.high - self.low)
        elif self.norm == 'minmax_neg': #min-max norm (-1 to 1)
            img_tensor = (img_tensor - (self.low + self.high) / 2) / ((self.high - self.low) / 2)
        elif self.norm == 'z_dataset': 
            img_tensor = (img_tensor - self.mean) / self.std
        elif self.norm == 'z_instance': 
            img_mean, img_std = img_tensor.mean(), img_tensor.std()
            img_tensor = (img_tensor - img_mean) / img_std

        if self.dataaug:
            transform = tio.Compose([tio.RandomFlip(axes=('LR', 'AP')), 
                                    tio.RandomAffine(degrees=(30, 30, 0)),
                                    tio.RandomNoise(0.05),
                                    ])
            img_tensor = transform(img_tensor)
        
        if self.acsconv:
            img_tensor = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])
        
        return img_tensor

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(os.listdir(self.img_dir))
