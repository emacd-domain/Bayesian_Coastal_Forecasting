# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:22:46 2024

@author: alphonse
"""

from numpy import random, repeat, floor, expand_dims

def repnoise(features, targets, n_times):
    copy_times = n_times#floor(n_times/features.shape[0])
    
    rep_features = repeat(features, copy_times, axis=0)
    rep_targets = repeat(targets, copy_times, axis=0)

    a1=random.uniform(low=0.99, high=1.01, size=len(rep_features))
    a1=repeat(expand_dims(a1,-1), features.shape[1], axis=1)
    a1=repeat(expand_dims(a1,-1), features.shape[2], axis=2)
    b1=random.uniform(low=-0.05, high=0.05, size=rep_features.shape)

    a2=random.uniform(low=0.99, high=1.01, size=len(rep_features))
    a2=repeat(expand_dims(a2,-1), targets.shape[1], axis=1)
    b2=random.uniform(low=-0.05, high=0.05, size=rep_targets.shape)

    return a1*rep_features+b1, a2*rep_targets+b2