#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:34:13 2022

@author: iason
Main training file for segmentation models on MaVeCoDD dataset.
TODO: Argparse parameters
"""

from eigennets import MaVeCoDD_dataset, MaVeCoDD
from eigennets import segmentation_models as sm
from sklearn.model_selection import KFold
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from torch.nn import BCELoss, Module
from torch.optim import Adam
import torch
from typing import Union, Iterable
import numpy as np

def _init_dataset() -> Union[Iterable, Iterable]:
    
    # Generate dataset
    dataset = MaVeCoDD()
    
    # Get train & label items
    train, label = dataset.get_dataset()
    
    return train, label

def _init_deterministic() -> str:
    
    # Set deterministic behaviour
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Cuda specific flags
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda'
    else:
    # Use cpu
        device = 'cpu'
    
    return device

def k_fold_validation(n_splits: int,
                      epochs: int,
                      network: str,
                      model_params: dict,
                      save_path: str='saved_models/segmentation_models/',
                      ) -> dict:
    # Number of k-fold splits
    n_splits = n_splits
    
    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    kf.get_n_splits(X=train, y=label)
    
    # K-Fold cross-validation iteration
    for i, (train_idx, test_idx) in enumerate(kf.split(X=train, y=label)):
        
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
        model = sm(network=network,
                   model_params=model_params)
        
        # Loss
        # loss = JaccardLoss(mode='binary')
        loss = BCELoss()
        
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
                                device=_device,
                                verbose=True,
                                )
        
        valid_epoch = ValidEpoch(model,
                                  loss=loss,
                                  metrics=metrics,
                                  device=_device,
                                  verbose=True,
                                )
        
        metrics = _training_loop(train_epoch=train_epoch,
                                 valid_epoch=valid_epoch,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 model=model,
                                 epochs=epochs,
                                 print_msg="\nEpoch: %d",
                                 )
        return metrics

def _training_loop(train_epoch: TrainEpoch,
                   valid_epoch: ValidEpoch,
                   train_loader: DataLoader,
                   valid_loader: DataLoader,
                   model: Module,
                   epochs: int,
                   print_msg: str="\nEpoch: %d",
                   ) -> dict:
    # Max score of loss
    max_score = 0.0
    
    # Keep track of metrics
    metrics = {'train_loss' : [],
               'valid_loss' : [],
               'train_metris' : [],
               'valid_metrics' : []}
    
    # Train model
    for epoch in range(epochs):
        # TODO: custom function ?
        print(print_msg)
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # Capture metrics
        metrics['train_loss'].append(train_logs['BCELoss'])
        metrics['valid_loss'].append(valid_logs['BCELoss'])
        metrics['train_metris'].append([train_logs['iou_score'], train_logs['fscore']])
        metrics['valid_metrics'].append([valid_logs['iou_score'], valid_logs['fscore']])
        
        # Save best model weights
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            # TODO: kwargs custom
            torch.save(model, f"{save_path}_{model.model_name}_best_model.pth")
            print('Model saved!')
    # Return metrics
    return metrics

if __name__ == "__main__":
    # Initialize cuda flags & enable
    _device = _init_deterministic()
    # Train & label samples
    train, label = _init_dataset()
    # Image resize
    target_size = (640, 640)
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
    network = 'unetplusplus'
    # Save path for models
    # TODO: Make a dynamic version as to not to re-write?
    save_path = 'saved_models/segmentation_models/'
    # Epochs
    epochs = 20
    # K-Fold splits
    n_split=4
    
    metrics = k_fold_validation(n_splits=n_split,
                                epochs=epochs,
                                network=network,
                                model_params=model_params,
                                save_path=save_path,
                                )