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

# %% Exercise 3, redone

def bacgro(t,c):
    
    kDA = k[0]
    kAD = k[1]
    kAS = k[2]
    kSA = k[3]
    
    cD = c[0]
    cA = c[1]
    cS = c[2]
    X = c[3]
    
    # adding given values
    vmax = 4
    kM = 0.715
    Y = 0.31
    b = 0.05
    
    # calculations
    mumax = Y*vmax
    dmMdt = vmax*X*(cD/(kM + cD))
    dXdt = ((mumax*cD)/(kM + cD))*X - b*X
    
    dmDdt = -kDA*cD + kAD*cA - (dmMdt)
    dmAdt = -kAS*cA + kSA*cS - kAD*cA + kDA*cD
    dmSdt = -kSA*cS + kAS*cA
    
    dbdt = X*b
    
    return [dmDdt, dmAdt, dmSdt, dXdt, dmMdt, dbdt]

# givens again
vmax = 4
kM = 0.715
Y = 0.31
b = 0.05
mumax = Y*vmax

inputs = [1, 0, 0, 0.01, 0, 0]
tstart = 0
tend = 60
k = (1.00, 0.50, 0.063201, 0.01)
output = solve_ivp(bacgro, [tstart,tend], inputs, method = 'RK45')
cD = output.y[0]
cA = output.y[1]
cS = output.y[2]
X = output.y[3] # biomass
mM = output.y[4]
db = output.y[5]
t = output.t
CO2 = (1-Y)*mM
#CO2 = (1-Y)*((cA + cS))

# when growth dX/dt = 0

# Mathematically, growth would be 0 when (mumaxcD/kM+cD) = b, or when the growth rate is equal to the death rate.
# solve (mumax * cD)/(kM + cD) - 0.05 = 0
# result (from calculator): cD = 0.030042 g

zerogrowthidx = np.argmax(np.abs(X))
zerogrowthtime = t[zerogrowthidx]
print(zerogrowthtime)
zerogrowthcD = cD[zerogrowthidx]
print(zerogrowthcD) # checks out

# how you can check if your code makes sense
# 1) numerical model in excel
# 2) calculate by hand if the value above makes sense--it does
# 3) ???

# %% Exercise 4

# part A -- plotting
plt.figure(3)
plt.plot(t,cD,label ='D')
plt.plot(t,cA,label ='A')
plt.plot(t,cS,label ='S')
plt.plot(t,X,label ='Biomass (X)')
plt.plot(t,CO2,label = 'CO2')
plt.legend(loc='best')
plt.xlabel('t [days]')
plt.ylabel('Mass (g)')
plt.title('Mass vs. time for reversible exchange simulation')
plt.grid()

# part B -- calculations
remaining = ((cD[-1] + cA[-1] + cS[-1])/1)*100
degraded = ((cA[-1] + cS[-1])/1)*100
Cinbiomass = (1-Y)*(cA[-1] + cS[-1])

# part C -- when the maximum X occurs
# growth dX/dt is the derivative of the amount of biomass X
# maximum X would occur where dX/dt = 0, which we just determined above
highestX = np.max(X)
print(highestX)
idxhighestX = np.where(X == highestX)
timeathighestX = t[idxhighestX]
print(timeathighestX) # time is the same, which checks out!

# part D -- when the maximum dX/Xdt occurs (growth rate)
# calculate algebraically: dX/Xdt = (mumax*cD)/(kM+cD) - b
# only variable in this is cD, and higher cD = higher dX/Xdt
# max dX/Xdt thus occurs at max cD

maxcD = np.max(cD)
print(maxcD)
idxmaxcD = np.where(cD == maxcD)
timeatmaxcD = t[idxmaxcD]
print(timeatmaxcD) # this makes sense logically

# %% Exercise 5
totalCend=cD[-1] + cA[-1] + cS[-1] + CO2[-1] + X[-1]
totalCstart=cD[0] + cA[0] + cS[0] + CO2[0] + X[0]
print(totalCstart - totalCend)
# The missing carbon is found in the dead material.
print(db[-1]) 

totalCend=cD[-1] + cA[-1] + cS[-1] + CO2[-1] + X[-1] + db[-1]
totalCstart=cD[0] + cA[0] + cS[0] + CO2[0] + X[0] + db[0]
print(totalCstart - totalCend)

# The reason that we have a difference of 0.01 (1.01 - 0.01) is because of the 
# initioal microbial degrader biomass X(0) (0.01) that is added to the 
# startconcencentration (1.00)

# %% Exercise 6

#a
#plotssss

#b
#When X(0)=0, biomass doesnt exist and the other parameters have to make up for it
#D and A does not decline as much and ends up being of a higher value than before
#but CO2 and S rises remarkably more than if X(0)=0.01

#c
#Of course, if Y and/or vmax equals 0, the growth rate is 0 and no new microbes will grow
#Which will result in death of all microbes almost instantly

#d
ratAD = cA[-1]/cD[-1]
ratSA = cS[-1]/cA[-1]
#they are not the same *values* are significantly higher,

# %% Exercise 7

cDcA = [0] * 5
allfit = [0] * 5
times = [2, 10, 18, 29, 40]
for i in range(len(times)):
    cDcA[i] = cD[times[i]] + cA[times[i]]
    allfit[i] = cD[times[i]] + cA[times[i]] + cA[times[i]] + X[times[i]] + db[times[i]]




