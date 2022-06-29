#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:48:24 2022

@author: iason

MaVeCoDD_dataset class file.

The implementation of the 'MaVeCoDD_dataset' class can serve as a generic
template for any vision/imaging purposes. It is assumed that minimal adaptation
is required.
"""

from torch import is_tensor
from torch.utils.data import Dataset
from numpy import arange, array, where, uint8
from numpy import sum as npsum
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

class MaVeCoDD_dataset(Dataset):
    
    def __init__(self,
                 train: list,
                 labels: list,
                 channels: int=3,
                 t_transforms: Compose = Compose([ToTensor(),
                                                  Normalize(
                                                      mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]
                                                      )
                                                  ]
                                                 ),
                 target_size: list = None,
                 l_transform: Compose = Compose([ToTensor()]),
                 ):
        
        self.input = train
        self.labels = labels
        self.channels = channels
        self.t_transforms = t_transforms
        self.target_size = target_size
        self.indexes = arange(len(train))
        self.l_transform = l_transform
        
        
    def __len__(self):
        
        return len(self.input)
    
    def __getitem__(self, idx):
        
        if is_tensor(idx):
            idx = idx.tolist()
        
        input_img_path = self.input[idx]
        target_img_path = self.labels[idx]
        # No target shape entered
        if self.target_size is None:
            temp = Image.open(input_img_path)
            self.target_size = temp.shape
        
        # Load training sample & corresponding label
        in_img = Image.open(input_img_path)
        label_img = Image.open(target_img_path)
        
        if self.target_size is not None:
            in_img = in_img.resize(self.target_size, Image.LANCZOS)
            label_img = label_img.resize(self.target_size, Image.LANCZOS)
        
        # Convert label to binary mask
        label_img = array(label_img)
        
        label_img = where(npsum(label_img, axis=2), 255, 0)
        
        # Convert back to PIL.Image for compatibility
        label_img = Image.fromarray(uint8(label_img), 'L')
        
        # Apply transformations
        x = self.t_transforms(in_img)
        y = self.l_transform(label_img)
        
        # Return training set & label(s)
        return x, y

# Deprecated code ---------------------------------------------------------- #
"""
Similar inner-workings with pytorch. No need for that much manual control
apparently.
"""
# -------------------------------------------------------------------------- #
    # def __getitem__(self, index: int) -> Union[tensor, tensor]:
    #     """Returns tuple (input, target) correspond to batch #idx."""
    #     shuffidx = index * self.batch_size
    #     idx = self.indexes[shuffidx * self.batch_size:
    #                        (shuffidx + 1) * self.batch_size]
        
    #     batch_input_img_paths = [self.input[k] for k in idx]
    #     batch_target_img_paths = [self.labels[k] for k in idx]
        
    #     # No target shape entered
    #     if self.target_size is None:
    #         # Batch size 1
    #         if self.batch_size == 1:
    #             temp = Image.open(batch_input_img_paths)
    #         else:
    #             temp = Image.open(batch_input_img_paths[0])
    #         self.target_size = temp.shape
    #     # Inputs and labels preallocation
        
    #     # Iterate over batch
    #     for i, (in_path, l_path) in enumerate(zip(batch_input_img_paths,
    #                                             batch_target_img_paths)):
    #         # Load training sample & corresponding label
    #         in_img = Image.open(in_path)
    #         label_img = Image.open(l_path)
            
    #         if self.target_size is not None:
    #             in_img = in_img.resize(self.target_size, Image.LANCZOS)
    #             label_img = label_img.resize(self.target_size, Image.LANCZOS)
            
    #         # Apply transformations
    #         x[i] = self.t_transforms(in_img)
    #         y[i] = self.l_transform(label_img)
            
    #     # Return training set & label(s)
    #     return x, y