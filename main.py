#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:34:13 2022

@author: iason
"""

from eigennets import _image_dataset, MaVeCoDD
from eigennets import segmentation_models as sm
from sklearn.model_selection import KFold
from segmentation_models_pytorch.losses import DiceLoss
# from segmentation_models_pytorch.metrics import f1_score, get_stats, iou_score
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Union, Iterable

def _init_dataset() -> Union[Iterable, Iterable]:
    
    # Generate dataset
    dataset = MaVeCoDD()
    
    # Get train & label items
    train, label = dataset.get_dataset()
    
    return train, label

DEVICE = 'cuda'

# def main() -> None:
# Train & label samples
train, label = _init_dataset()
# Image resize
target_size = (960, 960)
# Batch size
batch_size = 1

# Initialize model parameters - Segmentation models
model_params = {"encoder_name" : "resnet18",
                "encoder_depth" : 5,
                "encoder_weights" : None,
                "decoder_use_batchnorm" : True,
                "classes" : 1,
                "activation" : "sigmoid",
                }
epochs = 40

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

kf.get_n_splits(X=train, y=label)

# K-Fold cross-validation iteration
for i, (train_idx, test_idx) in enumerate(kf.split(X=train, y=label)):
    # Train dataset
    train_dataset = _image_dataset(train=train[train_idx],
                                   labels=label[train_idx],
                                   batch_size=batch_size,
                                   target_size=target_size)
    
    # Test dataset
    test_dataset = _image_dataset(train=train[test_idx],
                                   labels=label[test_idx],
                                   batch_size=batch_size,
                                   target_size=target_size)
    # Train dataloader
    train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)
    # Test dataloader
    valid_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=2)
    valid_loader.dataset[0]
    # Initialize model
    model = sm(model_params)
    model.unetplusplus()
    
    # Loss metric
    loss = DiceLoss(mode='binary')
    
    metrics = [IoU(threshold=0.5),
                Fscore(threshold=0.5)]
    
    optimizer = Adam(params=model.parameters(),
                      lr=0.0001)
    
    train_epoch = TrainEpoch(
                            model,
                            loss=loss,
                            metrics=metrics,
                            optimizer=optimizer,
                            device=DEVICE,
                            verbose=True,
                            )
    
    valid_epoch = ValidEpoch(model,
                              loss=loss,
                              metrics=metrics,
                              device=DEVICE,
                              verbose=True,
                            )
    # Train model
    for i in range(epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(test_dataset)
        

