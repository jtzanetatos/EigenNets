#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:34:13 2022

@author: iason
"""

from eigennets import _image_dataset, MaVeCoDD
from sklearn.model_selection import KFold

dataset = MaVeCoDD()

train, label =dataset.get_dataset() #np.asarray(train)[kspl[0][0].astype(int)]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

kf.get_n_splits(X=train, y=label)

kspl = []
for i, tt in enumerate(kf.split(X=train, y=label)):
    kspl.append(tt)
    print(i)