# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:14:36 2023

@author: mariu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from Step2 import theta,t_CSO,CSO_conc,CSOData,riverC,riverQ,EQS_exc
C0=0
os.chdir('C:/Users/mariu/Documents/Python Scripts/Environmental modelling/Assignment 1')
pd.options.mode.chained_assignment = None

from scipy.stats import lognorm, norm
x1=0.6
x2=5
p1=0.05
p2=0.95

x1=np.log(x1)
x2=np.log(x2)
p1ppf=norm.ppf(p1)
p2ppf=norm.ppf(p2)

scale = (x2 - x1) / (p2ppf - p1ppf)
mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)

Al_dist= lognorm(s=scale, scale=np.exp(mean))
Sample1e2 = np.random.lognormal(mean=mean, sigma=scale, size=int(1e2))

riverC[1]=0
riverC = riverC.reset_index(drop=True)

from Step2 import model

MC=10
MCsim=np.zeros([len(riverQ),MC])

for i in range(MC):
    out=model(riverC, riverQ, EQS_exc, CSOData, CSO_conc, C0)
    MCsim[:,i]=out['concentration']


    
CSO_conc=Al_dist.rvs(size=MC)
print(CSO_conc)
for i in range(MC):
    out=model(riverC, riverQ, EQS_exc, CSOData, CSO_conc[i], C0)
    MCsim[:,i]=out['concentration']


# CSO_flux=0
# selected_values = CSOData[CSOData['Vandmængd'] != 0]['Vandmængd']
# selected_values = selected_values.reset_index(drop=True)

# for i in range(1,len(selected_values)):
#     V_CSO=selected_values[i]/theta
#     Q_CSO=V_CSO/t_CSO
#     CSO_conc=Al_dist.rvs(size=MC)
#     CSO_flux=CSO_flux+Q_CSO*CSO_conc
    
    
    
q05=np.percentile(MCsim,5,axis=1)
q50=np.percentile(MCsim,50,axis=1)
q95=np.percentile(MCsim,95,axis=1)

t=MCsim[:,0]
plt.figure()
plt.rcParams.update({'font.size':20})

ax1=plt.subplot(2,1,1)
plt.plot(t,color='green',linestyle='-',label='deterministic')
plt.legend()
plt.grid()
ax1.set_xlabel('time [d]')
ax1.set_ylabel('Concentration [mg/l]')

ax2=plt.subplot(2,1,2)

for i in range (MC):
    plt.plot(MCsim[:,i],color=(0.7,0.7,0.7),linestyle='-')
plt.plot(q05,color='red',linestyle='--',label='5% percentile')
plt.plot(q50,color='blue',linestyle='-',label='median')
plt.plot(q95,color='red',linestyle='-',label='95% percentile')

plt.legend()
plt.grid()
plt.show()





    
