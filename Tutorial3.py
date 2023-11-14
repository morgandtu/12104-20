#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:02:26 2023

@author: benp.
"""

import pandas as pd
import numpy as np
from matplotlib import interactive
import matplotlib.pyplot as plt

# load measured values
measData = pd.read_csv('observed_values.csv',sep=';')
sim_time_steps = measData['Time']

# putting in values
par = np.zeros(3)
par[0] = 2.5e5 # value for k_inf [d^-1]
par[1] = 5e3 # value for E/R [K]
par[2] = .15 # value for initial concentration Sb [mg/L]

# calling function
import degradation_function as fun
out = fun.degradation_rate(sim_time_steps,par)

plt.figure(1)
#plot measurements
plt.plot(measData['Time'],measData['Concentration'],'.',color='red' )
plt.plot(sim_time_steps,out) #plot simulation data
plt.ylabel('concentration [mg/l]') #add y-label
plt.xlabel('time [s]') #add x-label
plt.grid(True) #add grid to graphs
plt.show()

# calling new function (RMSE)
import objective_functions as objFun
out2 = objFun.RMSE(measData['Concentration'],out)

# OAT analysis--taking par (middle) values, and making an array of 4 copies
parSample = np.tile(par,(4,1))

# adding 10% to each (∆theta/theta = 10%)--first row is original
dPar = .1
for i in range(3):
    parSample[i+1,i] = parSample[i+1,i]*(1+dPar)
    
# changing nothing
out0 = fun.degradation_rate(sim_time_steps, parSample[0,:])
out01 = out0[-1] # first output
out02 = objFun.RMSE(measData['Concentration'], out0)

# changing k inf
outA = fun.degradation_rate(sim_time_steps, parSample[1,:])
outA1 = outA[-1] # first output
outA2 = objFun.RMSE(measData['Concentration'], outA)

# changing E/R
outB = fun.degradation_rate(sim_time_steps, parSample[2,:])
outB1 = outB[-1] # first output
outB2 = objFun.RMSE(measData['Concentration'], outB)

# changing SB0
outC = fun.degradation_rate(sim_time_steps, parSample[3,:])
outC1 = outC[-1] # first output
outC2 = objFun.RMSE(measData['Concentration'], outC)

# sensitivity indices for each parameter
Si_lastConc = np.zeros(3)
Si_lastConc[0] = ((outA1-out01)/out01)/dPar
Si_lastConc[1] = ((outB1-out01)/out01)/dPar
Si_lastConc[2] = ((outC1-out01)/out01)/dPar
print(abs(Si_lastConc))

# B has the highest Si, so E/R is the most sensitive
plt.figure(2)
#plot measurements
plt.plot(measData['Time'], measData['Concentration'], '.', color ='red')
plt.plot(measData['Time'], out, label = 'original', color ='b')
plt.plot(measData['Time'], outA, label = '+10% k_inf', color ='orange')
plt.plot(measData['Time'], outB, label = '+10% E/R', color ='magenta')
plt.plot(measData['Time'], outC, label = '+10% SB0', color ='g')
plt.ylabel('concentration [mg/l]')
plt.xlabel('time [s]')
plt.legend()
plt.grid(True)
plt.show()

