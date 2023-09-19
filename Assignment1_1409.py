#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:16:15 2023

@author: benp.
"""

# importing
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ibtypetal = 1.7 # in mcg/L
ibpercentile = 1.8 # in mcg/L, using median to represent most common value

# %%
# =============================================================================
# STEP 2
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# housekeeping
#get_ipython().magic('reset -sf')
plt.close('all')

# importing data
riverData = pd.read_csv('riverNodes.csv', sep=",", decimal = ".")
CSOData = pd.read_csv('distances.csv', sep=",", decimal=".")

# data preparation
riverData = riverData[riverData['beregningspunktlokalid'].str.contains('MOELLEAA')] # selecting Moelle Aa data
riverData = riverData[riverData['aar']==2019] # selecting most recent data
riverData = riverData[riverData['maaned'].str.contains('januar')] # selecting highest flow data
# selecting upstream (outlet of lake) and downstream (outlet to sea) data
idxUp = riverData.index[riverData['beregningspunktlokalid'].str.contains('3687')][0]
# what this does is grab the row of riverData that contains that string, put it in a row, and take its index as an integer
idxDown = riverData.index[riverData['beregningspunktlokalid'].str.contains('13500')][0]
idxDiff = idxDown - idxUp # this is the # of river nodes we're examining

# selecting only the river data we need
riverData = riverData.loc[idxUp:idxDown,:]
riverData = riverData.reset_index(drop=True)

# initialize the variables needed by the model--I gave C one more column

riverQ = pd.DataFrame(np.zeros([idxDiff+1,6],dtype=float),columns=['X','Y','node ID','distance','flow','Qadded'])
riverC = pd.DataFrame(np.zeros([idxDiff+1,5],dtype=float),columns=['X','Y','node ID','distance','concentration'])
EQS_exc = pd.DataFrame(np.zeros([idxDiff+1,5],dtype=float),columns=['X','Y','node ID','distance','concentration'])

# adding data into columns
riverQ['X'] = riverData['X']
riverC['X'] = riverData['X']
riverQ['Y'] = riverData['Y']
riverC['Y'] = riverData['Y']
riverQ['node ID'] = riverData['beregningspunktlokalid']
riverC['node ID'] = riverData['beregningspunktlokalid']
riverQ['flow'] = riverData['vandfoering'].astype(float)
riverQ['Qadded'] = riverQ['Qadded'].astype(float)
riverQ['flow'] = riverQ['flow'].astype(float)
riverC['concentration'] = riverC['concentration'].astype(float)
EQS_exc['X'] = riverC['X']
EQS_exc['Y'] = riverC['Y']
EQS_exc['node ID'] = riverC['node ID']
EQS_exc['distance'] = riverC['distance']

# calculating distances
stringName = riverQ['node ID'].str.split('_').str[-1].astype(float)
riverQ['distance']= stringName - 3687
riverC['distance'] = riverQ['distance']

# the simple model: advection-dilution

# assigning values
riverC['concentration'][0] = 0.02
CSO_conc = 1.7*1000 # mcg/m^3
theta = 0.0231 # %
t_CSO = 4.3*3600 # seconds
EQS = 1700*1000 # mcg/m^3

# %%

# setting up empty matrix for CSOs
CSOname = pd.DataFrame(np.zeros([len(riverQ),1],dtype=float),columns=['name'])
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
            V_CSO = CSOData['Vandmaengd'][idxCSO[j]] * theta
            Q_CSO = V_CSO/t_CSO
            CSO_flux = CSO_flux + Q_CSO*CSO_conc
            CSO_Qtot = CSO_Qtot + Q_CSO
        riverQ['Qadded'][i] = riverQ['Qadded'][i-1] + CSO_Qtot
        riverC['concentration'][i] = (riverC['concentration'][i-1]*riverQ['flow'][i-1] + riverQ['Qadded'][i-1] + CSO_flux)/(riverQ['flow'][i] + riverQ['Qadded'][i])
    else:
        riverQ['Qadded'][i] = riverQ['Qadded'][i-1]
        riverC['concentration'][i] = riverC['concentration'][i-1]*(riverQ['flow'][i-1] + riverQ['Qadded'][i-1])/(riverQ[['flow'][i] + riverQ['Qadded'][i]])
riverC = riverC[riverC['concentration'] != 0]

EQS_exc['concentration'] = riverC['concentration'] > EQS
EQS_exc['X'] = riverC['X']
EQS_exc['Y'] = riverC['Y']
EQS_exc['node ID'] = riverC['node ID']
EQS_exc['distance'] = riverC['distance']
EQS_exc = EQS_exc.dropna(axis = 0)

plt.figure(1)
plt.plot(riverC['distance'],riverC['concentration'])
plt.xlabel('distance from the lake')
plt.ylabel('concentration of ibuprofen (mcg/m^3)')

plt.figure(2)
plt.plot(riverC['distance'],EQS_exc['concentration'])
plt.xlabel('distance from the lake')
plt.ylabel('concentration of ibuprofen (mcg/m^3))')


