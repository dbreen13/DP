#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:08:34 2024

@author: dbreen
"""
import numpy as np
import torch
import pickle

mean_CIFAR10 = [0.49139968, 0.48215841, 0.44653091]
std_CIFAR10 = [0.24703223, 0.24348513, 0.26158784]
batch=128
sizes=[[batch,64,4,4], [batch,128,4,4], [batch,256,4,4],[batch,512,4,4],[batch,64,2,2], [batch,64,8,8],[batch,64,16,16]]
for size in sizes:
    
    tens=np.random.normal(loc=mean_CIFAR10[0], scale=std_CIFAR10[0], size=size)
    tltens=torch.from_numpy(tens)
    print(size[1])
    with open(f'inch{size[1]}-wh{size[2]}.pkl', 'wb') as f: 
        pickle.dump(tltens, f)