#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:59:20 2023

@author: benp.
"""

import numpy as np

def test(obs_data,sim_data):
    objFun = 1
    return objFun

def RMSE(obs_data,sim_data):
    err = (sim_data-obs_data)**2 # squared error vector (vector)
    objFun = (np.mean(err))**.5 # calculate RMSE (a single number)
    return objFun

# new objective function
def invMSE(obs_data,sim_data):
    err=(sim_data-obs_data)**2 # error vector
    objFun=np.mean(err) # calculate MSE
    objFun=1/objFun # calculate inverse of MSE
    return objFun

# new objective function
def MSE(obs_data,sim_data):
    err=(sim_data-obs_data)**2 # error vector
    objFun=np.mean(err) # calculate MSE
    return objFun

def MARE(obs_data,sim_data):
    e = 0.5
    err = ((sim_data + e) - (obs_data + e))/(obs_data + e)
    objFun=abs(np.mean(err)) # calculate absolute mean relative error
    return objFun
