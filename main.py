#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:34:13 2022

@author: iason
"""

from eigennets import MaVeCoDD_dataset, MaVeCoDD
from eigennets import segmentation_models as sm
from sklearn.model_selection import KFold
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import seed as tseed
from torch import save as tsave
from torch.cuda import is_available
from typing import Union, Iterable
from os.path import join

def _init_dataset() -> Union[Iterable, Iterable]:
    
    # Generate dataset
    dataset = MaVeCoDD()
    
    # Get train & label items
    train, label = dataset.get_dataset()
    
    return train, label

# Torch seed
tseed(42)

# Use gpu if available
DEVICE = 'cuda' if is_available() else 'cpu'

# Train & label samples
train, label = _init_dataset()
# Image resize
target_size = (320, 320)
# Batch size
batch_size = 2

# Initialize model parameters - Segmentation models
model_params = {"encoder_name" : "resnet18",
                "encoder_depth" : 5,
                "encoder_weights" : None,
                "decoder_use_batchnorm" : True,
                "classes" : 1,
                "activation" : "sigmoid",
                }
# Save path for models
# TODO: Make a dynamic version as to not to re-write?
save_path = '/saved_models/segmentation_models/'
# Epochs
epochs = 20

# Number of k-fold splits
n_splits = 5

# Loss list
vk_loss = [ [] for _ in range(n_splits)]
tk_loss = vk_loss.copy()
# Metrics
tk_metris = tk_loss.copy()
vk_metris = tk_loss.copy()

# Initialize K-Fold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

kf.get_n_splits(X=train, y=label)

# K-Fold cross-validation iteration
for i, (train_idx, test_idx) in enumerate(kf.split(X=train, y=label)):
    # Max score of current split
    max_score = 0.0
    
    # Train dataset
    train_dataset = MaVeCoDD_dataset(train=train[train_idx],
                                   labels=label[train_idx],
                                   target_size=target_size)
    
    # Test dataset
    test_dataset = MaVeCoDD_dataset(train=train[test_idx],
                                   labels=label[test_idx],
                                   target_size=target_size)
    # Train dataloader
    train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)
    # Test dataloader
    valid_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    
    # Initialize model
    model = sm(model_params)
    model.unetplusplus()
    
    # Loss
    loss = JaccardLoss(mode='binary')
    
    # Metrics
    metrics = [IoU(threshold=0.5),
               Fscore(threshold=0.5)]
    
    # Optimizer
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
    for epoch in range(epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # Capture metrics
        vk_loss[i].append(valid_logs['JaccardLoss'])
        tk_loss[i].append(train_logs['JaccardLoss'])
        tk_metris[i].append(train_logs['iou_score'], train_logs['fscore'])
        vk_metris[i].append([valid_logs['iou_score'], valid_logs['fscore']])
        
        # Save best model weights
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            tsave(model, join(save_path, f"k_split{i+1}_{vk_loss[i][epoch]}_iou{max_score}_model.pth"))
            print('Model saved!')

