# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:14:36 2023

@author: mariu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from Step2 import theta,CSOname,t_CSO,CSO_conc,CSOData,riverC,riverQ

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
Sample1e2 = Al_dist.rvs(size=int(1e2))
Sample1e3 = Al_dist.rvs(size=int(1e3))
Sample1e4 = Al_dist.rvs(size=int(1e4))
print('%0.f,%0.f,%0.f' % (np.mean(Sample1e2),np.percentile(Sample1e2,25),np.percentile(Sample1e2,77)))
print('%0.f,%0.f,%0.f' % (np.mean(Sample1e2),np.percentile(Sample1e3,25),np.percentile(Sample1e3,77)))
print('%0.f,%0.f,%0.f' % (np.mean(Sample1e2),np.percentile(Sample1e4,25),np.percentile(Sample1e4,77)))
Sample1e2 = np.random.lognormal(mean=mean, sigma=scale, size=int(1e2))

riverC[1]=0
def riverModel(riverQ,CSOData,CSO_conc):
    for i in range(1,len(CSOData)): # looping through CSO dataframe
        CSOSearch = riverQ['node ID'].str.contains(CSOData['HubName'][i]) # selecting for where any node ID in riverQ matches the i-th hubname in CSO
        isThereCSO = CSOSearch.sum() == 1 # setting a boolean to 1 if there is a CSO
        if isThereCSO:
            CSOname['name'][i] = riverQ["node ID"].iloc[i] # assigning the names of the matching CSOs to a dataframe
        idxCSO = (CSOname[CSOname['name'] != 0]).index # getting the indices **in CSOData, not in riverQ!!!**
        # running the model--all of this is from the pseudocode
        if len(idxCSO) > 0:
            CSO_flux = 0
            CSO_Qtot = 0
            for j in range(len(idxCSO)):
                V_CSO = CSOData['Vandmængd'][idxCSO[j]] * theta
                Q_CSO = V_CSO/t_CSO
                CSO_flux = CSO_flux + Q_CSO*CSO_conc
                CSO_Qtot = CSO_Qtot + Q_CSO
            riverQ['Qadded'][i] = riverQ['Qadded'][i-1] + CSO_Qtot
            riverC['concentration'][i] = (riverC['concentration'][i-1]*riverQ['flow'][i-1] + riverQ['Qadded'][i-1] + CSO_flux)/(riverQ['flow'][i] + riverQ['Qadded'][i])
        else:
            riverQ['Qadded'][i] = riverQ['Qadded'][i-1]
            riverC['concentration'][i] = riverC['concentration'][i-1]*(riverQ['flow'][i-1] + riverQ['Qadded'][i-1])/(riverQ[['flow'][i] + riverQ['Qadded'][i]])
    return riverC


MC=10

MCsim=np.zeros([44,MC])
for i in range(1,MC):
    out=riverModel(riverQ,CSOData,CSO_conc)
    MCsim[:,i]=out['conc']




    
