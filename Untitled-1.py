
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def mmmonod(t,c):
    kDA = k[0]
    kAD = k[1]
    kAS = k[2]
    kSA = k[3]
    
    mD = c[0]
    mA = c[1]
    mS = c[2]
    X  = c[3]
    
    # table 3 input data
    vmax = 4 # g C substrate
    Km = 0.715 # g C m^-3
    Y = 0.31 # g C biomass
    b = 0.05 # d^-1
    
    # for exercise 3
    dmMdt = vmax*X*(mD/(Km+mD))
    mumax = Y*vmax
    dXdt = ((mumax*mD)/(Km+mD))*X-b*X
    
    # part a
    dmAdt = -kAS*mA + kSA*mS - kAD*mA + kDA*mD
    dmSdt = -kSA*mS + kAS*mA
    dmDdt = -kDA*mD + kAD*mA - (dmMdt + dXdt)
    
    return [dmDdt, dmAdt, dmSdt, dXdt]

x0 = [1,0,0,0.01] # mD, mA, mS, and X
tstart = 0
tend = 60
k = (1.00, 0.50, 0.063201, 0.01) # defining k values
t_eval = np.linspace(tstart, tend, num=10000)
xr = solve_ivp(mmmonod,[tstart,tend],x0, method ='RK45', t_eval = t_eval) # calling function

# assigning values from functions
Dr = xr.y[0]
Xr = xr.y[3]
