#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:15:20 2022

@author: iason
"""

from segmentation_models_pytorch import (UnetPlusPlus, PSPNet, FPN,
                                         DeepLabV3Plus)
from torch import save, load
from torch.nn import Module, Sequential
from typing import Iterable

def _forward(obj: object, x: Iterable):
    return obj.layers(x)

class segmentation_models(Module):
    
    def __init__(self,
                 model_params: dict,
                 custom_train: bool=False,
                 layer: Sequential=None,
                 ):
        super(segmentation_models, self).__init__()
        self.model_params = model_params
        # Initialize parameters for the models
        self._init_params(model_params)
        # Further experiment with architecture
        if custom_train:
            self._custom_train = custom_train
            # Set forward method
            setattr(self, 'forward', _forward)
            setattr(self, 'layers', layer)
    
    def _init_params(self, model_params: dict):
        # Iterate over keys and values
        for key in list(model_params.keys()):
            setattr(self, key, model_params[key])
    
    def _base_model(self, model):
        
        self.model = model(**self.model_params)
        
        attrs = [a for a in dir(self.model) if not a.startswith("_")]
        attrs = attrs[1:]
        
        for attr in attrs:
            setattr(self, attr, eval(f"self.model.{attr}"))
    
    def unetplusplus(self):
        # Initialize model
        self._base_model(UnetPlusPlus)
        self.model_name = 'UnetPlusPlus'
    
    def pspnet(self):
        self._base_model(PSPNet)
        self.model_name = 'PSPNet'
        
    def fpn(self):
        self._base_model(FPN)
        self.model_name = "FPN"
        
    def deeplabv3p(self):
        self._base_model(DeepLabV3Plus)
        self.model_name = 'Deeplabv3+'
    
    def saveWeights(self):
        save(self.model.state_dict(), f"./pytorch_{self.model_name}_weights.pth")
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.model.load_state_dict(load(f"./pytorch_{self.model_name}_weights.pth"))
        print("Weights loaded successfuly.")

if __name__ == "__main__":
    model_params = {"encoder_name" : "resnet34",
                    "encoder_depth" : 5,
                    "encoder_weights" : 'imagenet',
                    "decoder_use_batchnorm" : True,
                    }
    model = segmentation_models(model_params)
    model.unetplusplus()