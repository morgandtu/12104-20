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

# # housekeeping
# get_ipython().magic('reset -sf')
# plt.close('all')

# # %%
# # =============================================================================
# # STEP 1
# # =============================================================================

# # %% --- TASK 1.1 ---

# # getting data
# riverNodes = pd.read_csv('riverNodes.csv', sep=",", decimal = ".")
# distances = pd.read_csv('distances.csv', sep=",", decimal=".")

# # removing unnecessary data
# riverNodes.rename(columns={"vandfoering": "water flow (m^3/s)"}, inplace=True)
# distances.rename(columns={"BI-5 (kg)": "Biological oxygen demand, 5 days (kg)"}, inplace=True)
# distances = distances.drop(['Reduceret'],axis=1)
# distances = distances.drop(['Udlednings'],axis=1)
# distances = distances.drop(['Bygvaerkst'],axis=1)
# distances = distances.drop(['Godkendels'],axis=1)
# distances = distances.drop(['Ejer'],axis=1)
# distances = distances.drop(['Idriftsat'],axis=1)
# distances = distances.drop(['Nedlagt'],axis=1)
# distances = distances.drop(['Antal over'],axis=1)

# meandistm = distances.loc[:,'HubDist'].mean()*1000

# # There are 27 CSO structures along the stream, located along the banks of the stream from 
# # the outlet to Lyngby Lake to the Oeresund. They are located, on average, 103 m
# # from the banks of the stream. 

# annualCSOwaterflow = distances.loc[:,'Vandmaengd'].sum()
# pollutants = distances.iloc[:,7:10].sum()
# pollutants = pollutants.sum()

# # 289666 m^3 of water and 12744 kg of pollutants were discharged into the stream. 
# # The UF01001 CSO in Ravnholm saw the
# # largest number of overflows, and the Ly R16 CSO saw the largest volume of water. 
# # Ly R16 is located close to a bridge near the outlet to Lyngby Lake, while UF01001 
# # is located near a highway overpass, closer to the ocean. We would thus expect the
# # major negative effects to occur closer to the ocean, where pollutants have
# # accumulated as they have flowed through the stream. 

# meanflow = riverNodes.loc[:,'water flow (m^3/s)'].mean()
# months = ['januar','februar','marts','april','maj','juni','juli','august','september','oktober','november','december']

# # looping through the month column, pulling out values for each month, 
# # calculating the average, and then putting it in the empty array I made
# monthflows=pd.DataFrame(np.zeros([12,2],dtype=float),columns=['month','flow (m^3/s)'])
# for i in range(len(months)):
#     monthflows['month'][i] = months[i]
#     idx = riverNodes['maaned'].str.contains(months[i])
#     data = riverNodes[idx]
#     data = data.reset_index(drop=True)
#     monthflows['flow (m^3/s)'][i] = data.loc[:,'water flow (m^3/s)'].mean()

# # The average flow of the river in Moelle Aa is 0.24 m3/s. Highest month is Jan, lowest August

# # idxflowLyngby = riverNodes['beregningspunktlokalid'].str.contains('Novana_Model_MOELLEAA_DK1_3687')
# # flowLyngby = riverNodes[idxflowLyngby]
# # flowLyngby = flowLyngby.reset_index(drop=True)

# # years = list(range(1990, 2020))
# # yearsLyngby=pd.DataFrame(np.zeros([30,2],dtype=float),columns=['year','flow (m^3/s)'])
# # for i in range(len(years)):
# #     yearsLyngby['year'][i] = years[i]
# #     idx = flowLyngby['aar'] == years[i]
# #     data = flowLyngby[idx]
# #     data = data.reset_index(drop=True)
# #     yearsLyngby['flow (m^3/s)'][i] = data.loc[:,'water flow (m^3/s)'].sum()
# #     averageannual = yearsLyngby['flow (m^3/s)'].mean()

# # %%  --- TASK 1.2 ---

# # importing in annual data and selecting for the outlet from the lake
# riverAnnual = pd.read_csv('yearlyvalues.csv', sep=",", decimal = ".")
# riverAnnual = riverAnnual.tail(-1)
# idxflowLyngby2 = riverAnnual['beregningspunktlokalid'].str.contains('Novana_Model_MOELLEAA_DK1_13500')
# flowLyngby2 = riverAnnual[idxflowLyngby2]
# flowLyngby2 = flowLyngby2.reset_index(drop=True)

# # finding the average annual flow in m^3/s and convering to m^3/year
# avgyear = flowLyngby2['vandfoering'].mean()*3600*24*365
# fraction = annualCSOwaterflow/avgyear # 1.26% of the water entering the Oeresund is coming from the CSOs

# # getting nitrogen data into Python for Lyngby lake
# lakeData = pd.DataFrame(np.zeros([12,2],dtype=float),columns=['year','nitrogen (mg/L)'])
# years = list(range(2002, 2015))
# nmgL = [0.07, 0.0155, 0.02, 0.15, 0.03, 0.04, 0.173, 0.25, 0.01, 0.3, 0.15, 0.075, 0]
# for i in range(len(years)):
#      lakeData['year'][i] = years[i]
#      lakeData['nitrogen (mg/L)'][i] = nmgL[i]

# # converting the kg of nitrogen passing through the CSOs to mg/L
# CSOnitromg = distances.loc[:,"Total-N (k"].sum()*1000000
# CSOmgnitroperL = CSOnitromg/(annualCSOwaterflow*1000)

# # looping through CSO dataset to find nitrogen contribution from each CSO
# CSOnitro = pd.DataFrame(np.zeros([27,2],dtype=float),columns=['CSO','nitrogen (mg/L)'])
# for i in range(len(distances)):
#     CSOnitro['CSO'][i] = distances['Navn'][i]
#     kgnitro = distances['Total-N (k'][i]
#     mgnitro = kgnitro*1000000
#     Lflow = distances['Vandmaengd'][i]*1000
#     CSOnitro['nitrogen (mg/L)'][i] = mgnitro/Lflow

# # major source of nitrogen is CSOs and we know the specific ones. 

# %% --- TASK 1.3 ---

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
CSO_conc = 1.7*(10**(-6)) # g/L
theta = 2.31 # %
t_CSO = 4.3 # hr
EQS = 1700*(10**(-6)) # g/L

# %% IGNORE THIS!!!
# the model
# getting CSO indices

# CSOname = pd.DataFrame(np.zeros([len(riverQ),1],dtype=float),columns=['name'])
# for i in range(1,idxDiff):
#     CSOSearch = CSOData['HubName'].str.contains(riverQ['node ID'][i])
#     isThereCSO = CSOSearch.sum() == 1
#     if isThereCSO:
#         CSOname['name'][i] = riverQ["node ID"].iloc[i]
#     idxCSO = (CSOname[CSOname['name'] != 0]).index
#     # running the model
#     if len(idxCSO) > 0:
#         CSO_flux = 0
#         CSO_Qtot = 0
#         for j in range(len(idxCSO)):
#             V_CSO = CSOData['Vandmaengd'][idxCSO[j]] * theta
#             Q_CSO = V_CSO/t_CSO
#             CSO_flux = CSO_flux + Q_CSO*CSO_conc
#             CSO_Qtot = CSO_Qtot + Q_CSO
#         riverQ['Qadded'][i] = riverQ['Qadded'][i-1] + CSO_Qtot
#         riverC['concentration'][i] = (riverC['concentration'][i-1]*riverQ['flow'][i-1] + riverQ['Qadded'][i-1] + CSO_flux)/(riverQ['flow'][i] + riverQ['Qadded'][i])
#     else:
#         riverQ['Qadded'][i] = riverQ['Qadded'][i-1]
#         riverC['concentration'][i] = riverC['concentration'][i-1]*(riverQ['flow'][i-1] + riverQ['Qadded'][i-1])/(riverQ[['flow'][i] + riverQ['Qadded'][i]])
# EQS_exc['concentration'] = riverC['concentration'] > EQS


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
plt.ylabel('concentration of ibuprofen (g/L)')

plt.figure(2)
plt.plot(riverC['distance'],EQS_exc['concentration'])
plt.xlabel('distance from the lake')
plt.ylabel('concentration of ibuprofen (g/L)')


