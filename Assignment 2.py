# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:52:34 2023

@author: mariu
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os
os.chdir('C:/Users/mariu/Documents/Python Scripts/Environmental modelling/Assignment 2')

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

# %% Exercise 3
def reversible(t,c):
    kDA = k[0]
    kAD = k[1]
    kAS = k[2]
    kSA = k[3]
    
    mD = c[0]
    mA = c[1]
    mS = c[2]
    
    X  = c[3]
    cD = c[4]
    
    vMax = 4
    kM = 0.715
    Y = 0.31
    b = 0.05
    
    # part a
    dmDdt = -kDA*mD + kAD*mA
    dmAdt = -kAS*mA + kSA*mS - kAD*mA + kDA*mD
    dmSdt = -kSA*mS + kAS*mA
    
    # Exercise 3
    muMax = Y * vMax   
    dmMdt = vMax*X*(cD/(kM+cD))
    dXdt  = ((muMax*cD)/(kM+cD))-b*X
    return [dmDdt, dmAdt, dmSdt, dmMdt, dXdt]

x0 = [1,0,0,0.01,1] # initial conditions
tstart = 0
tend= 60
k = (1.00, 0.50, 0.063201, 0.01) # defining k values
t_eval = np.linspace(tstart, tend, num=10000)
xr = solve_ivp(reversible,[tstart,tend],x0, method ='RK45', t_eval=t_eval) # calling function

gr = xr.y[4]
t = xr.t

# Find the index where growth rate is closest to zero
zero_growth_index = np.argmin(np.abs(gr))
print(zero_growth_index)

# Get the time and concentration of cD at that index
time_at_zero_growth = t[zero_growth_index]
cD_at_zero_growth = xr.y[3, zero_growth_index]  # cD is the fourth component




