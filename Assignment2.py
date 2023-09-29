#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:25:34 2023

@author: benp.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# %% part 1

# exercise 1 

# functions for reversible and irreversible exchange
def reversible(t,c):
    kDA = k[0]
    kAD = k[1]
    kAS = k[2]
    kSA = k[3]
    
    mD = c[0]
    mA = c[1]
    mS = c[2]
    # part a
    dmDdt = -kDA*mD + kAD*mA
    dmAdt = -kAS*mA + kSA*mS - kAD*mA + kDA*mD
    dmSdt = -kSA*mS + kAS*mA
    return [dmDdt, dmAdt, dmSdt]

def irreversible(t,c):
    kDA = k[0]
    kAD = k[1]
    kAS = k[2]
    kSA = k[3]
    # part a
    mD = c[0]
    mA = c[1]
    mS = c[2]
    
    dmDdt = -kDA*mD
    dmAdt = -kAS*mA + kDA*mD
    dmSdt = kAS*mA
    return [dmDdt, dmAdt, dmSdt]

x0 = [1,0,0] # initial conditions
tstart = 0 # simulation length
tend = 12
k = (1.00, 0.50, 0.063201, 0.01) # defining k values
xr = solve_ivp(reversible,[tstart,tend],x0, method ='RK45') # calling function
xi = solve_ivp(irreversible,[tstart,tend],x0, method ='RK45') # calling function

# assigning values from functions
Dr = xr.y[0]
Ar = xr.y[1]
Sr = xr.y[2]
Di = xi.y[0]
Ai = xi.y[1]
Si = xi.y[2]

# plotting
# part b
plt.figure(1)
plt.plot(xr.t,Dr,label ='D')
plt.plot(xr.t,Ar,label ='A')
plt.plot(xr.t,Sr,label ='S')
plt.legend(loc='best')
plt.xlabel('t [days]')
plt.ylabel('Concentration')
plt.title('Mass vs. time for reversible exchange')
plt.grid()

# part c
plt.figure(2)
plt.plot(xi.t,Di,label ='D')
plt.plot(xi.t,Ai,label ='A')
plt.plot(xi.t,Si,label ='S')
plt.legend(loc='best')
plt.xlabel('t [days]')
plt.ylabel('Concentration')
plt.title('Concentration vs. time for irreversible exchange')
plt.grid()

# part d
CaCd = Ar[-1]/Dr[-1] # stabilizes at 1.999 (2.00)
CsCa = Sr[-1]/Ar[-1] # stabilizes at 6.32

# exercise 2







