#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:48:24 2022

@author: iason
"""

from typing import Union
from torch import zeros, tensor
from torch.utils.data import DataLoader
from warnings import warn
from numpy import arange
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

class _image_dataset(DataLoader):
    
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
                 batch_size: int = 1,
                 target_size: list = None,
                 l_transform: Compose = Compose([ToTensor()]),
                 ):
        if batch_size < 1:
            warn("Batch size cannot be less than 1, setting size to 1.")
            batch_size = 1
        
        self.input = train
        self.labels = labels
        self.channels = channels
        self.t_transforms = t_transforms
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = arange(len(train))
        self.l_transform = l_transform
        
        
    def __len__(self):
        
        return len(self.input) // self.batch_size
    
    def __getitem__(self, index: int) -> Union[tensor, tensor]:
        """Returns tuple (input, target) correspond to batch #idx."""
        shuffidx = index * self.batch_size
        idx = self.indexes[shuffidx * self.batch_size:
                           (shuffidx + 1) * self.batch_size]
        
        batch_input_img_paths = [self.input[k] for k in idx]
        batch_target_img_paths = [self.labels[k] for k in idx]
        
        # No target shape entered
        if self.target_size is None:
            # Batch size 1
            if self.batch_size == 1:
                temp = Image.open(batch_input_img_paths)
            else:
                temp = Image.open(batch_input_img_paths[0])
            self.target_size = temp.shape
        # Inputs and labels preallocation
        # x = zeros(((self.batch_size,) + (self.channels,) + (self.target_size)))
        # y = zeros(((self.batch_size,) + (self.channels,) + (self.target_size)))
        x = zeros(((self.channels,) + (self.target_size)))
        y = zeros(((self.channels,) + (self.target_size)))
        
        # Iterate over batch
        for i, (in_path, l_path) in enumerate(zip(batch_input_img_paths,
                                                batch_target_img_paths)):
            # Load training sample & corresponding label
            in_img = Image.open(in_path)
            label_img = Image.open(l_path)
            
            if self.target_size is not None:
                in_img = in_img.resize(self.target_size, Image.LANCZOS)
                label_img = label_img.resize(self.target_size, Image.LANCZOS)
            
            # Apply transformations
            # x[i] = self.t_transforms(in_img)
            # y[i] = self.l_transform(label_img)
            x = self.t_transforms(in_img)
            y = self.l_transform(label_img)
        # Return training set & label(s)
        return x, y