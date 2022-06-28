#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:46:13 2022

@author: iason
"""

from os import listdir
from os.path import join, abspath
from typing import TypedDict, List
from warnings import warn
from numpy import asarray

class MaVeCoDD_dtype(TypedDict):
    split: str
    dataset: tuple

def _split_safeguard(obj: object) -> None:
    # Key invalid, set default behaviour
    if obj.split_type not in obj._res_split:
        warn("Invalid key entered, default behaviour is mixed data mining",
             category=SyntaxWarning)
        obj.split_type = "mixed"

def _list2array(dirs: list) -> None:
    # Initialize new list
    dataset = []
    
    # Iterate over dirs
    for path in dirs:
        dataset.append(asarray(path))
    
    return dataset


class MaVeCoDD:
    
    def __init__(self, split_type: str = "mixed", root: str = "./MaVeCoDD"):
        
        self.split_type = split_type
        self.root = root
        # TODO: Set outside init?
        self._res_split: MaVeCoDD_dtype = {"mixed" : (('HiRes', 'LoRes'),
                                                     ('HiReslabels', 'LoReslabels')),
                                          "highres" : (('HiRes'),
                                                       ('HiReslabels')),
                                          "lores" : (('LoRes'),
                                                     ('LoReslabels')),
                                          }
        self.dataset_split = self._dataset_split()
        self.dataset = self._load_mavecodd()
    
    def _dataset_split(self):
        _split_safeguard(self)
        return self._res_split[self.split_type]
        
    # TODO: Parallel access to directories
    def _set_subdir_paths(self) -> List:
        
        # Initialize dataset
        dataset = [[], []]
        # Mixed case
        if self.split_type == "mixed":
            # Iterate over training/labeling set of dirs
            for i, img_set in enumerate(self.dataset_split):
                # Iterate over current set of dirs
                for path in img_set:
                    # Iterate over contents of current dir
                    for img in listdir(join(self.root, path)):
                        dataset[i].append(abspath(join(self.root,
                                                       join(path, img))))
        # Other cases
        else:
            # Iterate over train/label set of dirs
            for i, path in enumerate(self.dataset_split):
                # Iterate over contents of current dir
                for img in listdir(join(self.root, path)):
                    dataset[i].append(abspath(join(self.root,
                                                   join(path, img))))
        # Sort paths to ensure proper sequence
        for i in range(len(dataset)):
            dataset[i].sort()
        
        # Convert to array
        dataset = _list2array(dataset)
        # Return dataset
        return dataset
    
    def _load_mavecodd(self):
        return self._set_subdir_paths
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def get_dataset(self):
        return self.dataset



# Deprecated --------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# def _merge_mixed(dirs: list) -> None:
#     # Size of dataset
#     d_size = len(dirs[0])
#     # Initialize list of train/label arrays
#     dataset = [zeros(d_size, dtype=str) for _ in range(2)]
    
#     # Sub-bucket indices
#     l_idx = 0
#     t_idx = 0
    
#     # Iterate over dataset
#     for i, dir_imgs in enumerate(dirs):
#         # Label bucket
#         if dir_imgs[0].find("label") != -1:
#             # First sub-bucket
#             if l_idx == 0:
#                 dataset[1][:d_size] = asarray(dir_imgs).copy()
#                 # Update index
#                 l_idx += 1
#             else:
#                 dataset[1][d_size:] = asarray(dir_imgs).copy()
#         # Train bucket
#         else:
#             # First sub-bucket
#             if t_idx == 0:
#                 dataset[0][:d_size] = asarray(dir_imgs).copy()
#                 # Update index
#                 t_idx += 1
#             else:
#                 dataset[0][d_size:] = asarray(dir_imgs).copy()
#     # Return merged dataset
#     return dataset