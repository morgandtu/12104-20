#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:51:56 2023

@author: benp.
"""

# ASSIGNMENT 1: SIMPLE POLLUTANT MASS BALANCES TO ASSESS THE EMISSIONS OF COMBINED SEWER OVERFLOW INTO THE MØLLE Å STREAM

# importing
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% task 1.1

# getting data
riverData = pd.read_csv('riverNodes.csv', sep=",", decimal = ".")
CSOData = pd.read_csv('distances.csv', sep=",", decimal=".")

# removing unnecessary data
CSOData = CSOData.drop(['Zgeometry'], axis=1)
CSOData = CSOData.drop(['Reduceret'], axis=1)
CSOData = CSOData.drop(['Bassinvolu'], axis=1)
CSOData = CSOData.drop(['Udlednings'], axis=1)
CSOData = CSOData.drop(['Bygværkst'], axis=1)
CSOData = CSOData.drop(['Godkendels'], axis=1)
CSOData = CSOData.drop(['Ejer'], axis=1)
CSOData = CSOData.drop(['Idriftsat'], axis=1)
CSOData = CSOData.drop(['Nedlagt'], axis=1)
CSOData = CSOData.drop(['Antal over'], axis=1)

# finding mean distances in m from CSO
meandistm = CSOData.loc[:,'HubDist'].mean()*1000
# finding annual water flow in m^3
annualCSOwaterflow = CSOData.loc[:,'Vandmængd'].sum()
# finding pollutants in kg
pollutants = CSOData.iloc[:,5:8].sum()
pollutants = pollutants.sum()

meanflow = riverData.loc[:,'vandfoering'].mean()
months = ['januar','februar','marts','april','maj','juni','juli','august','september','oktober','november','december']

# looping through the month column, pulling out values for each month, 
# calculating the average, and then putting it in the empty array I made
monthflows=pd.DataFrame(np.zeros([12,2],dtype=float),columns=['month','flow (m^3/s)'])
for i in range(len(months)):
    monthflows['month'][i] = months[i]
    idx = riverData['maaned'].str.contains(months[i])
    data = riverData[idx]
    data = data.reset_index(drop=True)
    monthflows['flow (m^3/s)'][i] = data.loc[:,'vandfoering'].mean()

# %% task 1.2

# importing in annual data and selecting for the outlet from the lake
riverAnnual = pd.read_csv('yearlyvalues.csv', sep=",", decimal = ".")
riverAnnual = riverAnnual.tail(-1)
idxflowLyngby2 = riverAnnual['beregningspunktlokalid'].str.contains('Novana_Model_MOELLEAA_DK1_13500')
flowLyngby2 = riverAnnual[idxflowLyngby2]
flowLyngby2 = flowLyngby2.reset_index(drop=True)

# finding the average annual flow in m^3/s and convering to m^3/year
avgyear = flowLyngby2['vandfoering'].mean()*3600*24*365
fraction = annualCSOwaterflow/avgyear # 1.26% of the water entering the Oeresund is coming from the CSOs

# getting nitrogen data into Python for Lyngby lake
lakeData = pd.DataFrame(np.zeros([12,2],dtype=float),columns=['year','nitrogen (mg/L)'])
years = list(range(2002, 2015))
nmgL = [0.07, 0.0155, 0.02, 0.15, 0.03, 0.04, 0.173, 0.25, 0.01, 0.3, 0.15, 0.075, 0]
for i in range(len(years)):
      lakeData['year'][i] = years[i]
      lakeData['nitrogen (mg/L)'][i] = nmgL[i]

# converting the kg of nitrogen passing through the CSOs to mg/L
CSOnitromg = CSOData.loc[:,"Total-N (k"].sum()*1000000
CSOmgnitroperL = CSOnitromg/(annualCSOwaterflow*1000)

# looping through CSO dataset to find nitrogen contribution from each CSO
CSOnitro = pd.DataFrame(np.zeros([27,2],dtype=float),columns=['CSO','nitrogen (mg/L)'])
for i in range(len(CSOData)):
    CSOnitro['CSO'][i] = CSOData['Navn'][i]
    kgnitro = CSOData['Total-N (k'][i]
    mgnitro = kgnitro*1000000
    Lflow = CSOData['Vandmængd'][i]*1000
    CSOnitro['nitrogen (mg/L)'][i] = mgnitro/Lflow

# %% task 1.3

ibtypetal = 1.7 # in mcg/L
ibpercentile = 1.8 # in mcg/L, using median to represent most common value

# %% task 2.2

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
riverQ['node ID'] = riverQ['node ID'].astype(str)
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
for i in range (0,len(riverC)):
    stringName = riverQ['node ID'][i].split('_')
    riverQ['distance'][i] = float(stringName[4]) - 3687
riverC['distance'] = riverQ['distance']


# the simple model: advection-dilution

#  assigning values
riverC['concentration'][0] = 0
CSO_conc = 1.7*1000 # mcg/m^3
theta = 0.0231 # %
t_CSO = 4.3*3600 # seconds
EQS = 1700*1000 # mcg/m^3
x = 0
count = 0

# calculations
for i in range (1,len(riverQ)):
    idxCSO = CSOData['HubName'].str.contains(riverQ['node ID'][i])
    idxCSO = idxCSO.index[idxCSO == True].tolist()
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
        riverC['concentration'][i] = riverC['concentration'][i-1] * (riverQ['flow'][i-1] * riverQ['Qadded'][i-1])/(riverQ['flow'][i] + riverQ['Qadded'][i])

EQS_exc['concentration'] = riverC['concentration'] > EQS
EQS_exc['X'] = riverC['X']
EQS_exc['Y'] = riverC['Y']
EQS_exc['node ID'] = riverC['node ID']
EQS_exc['distance'] = riverC['distance']
EQS_exc = EQS_exc.dropna(axis = 0)

# %% function for model

def model(riverC, riverQ, EQS_exc, CSOData, CSO_conc, C0):
    # assigning values
    riverC['concentration'][0] = C0
    CSO_conc = CSO_conc # mcg/m^3
    theta = 0.0231 # %
    t_CSO = 4.3*3600 # seconds
    EQS = 1700*1000 # mcg/m^3

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
                V_CSO = CSOData['Vandmængd'][idxCSO[j]] * theta
                Q_CSO = V_CSO/t_CSO
                CSO_flux = CSO_flux + Q_CSO*CSO_conc
                CSO_Qtot = CSO_Qtot + Q_CSO
            riverQ['Qadded'][i] = riverQ['Qadded'][i-1] + CSO_Qtot
            riverC['concentration'][i] = (riverC['concentration'][i-1]*riverQ['flow'][i-1] + riverQ['Qadded'][i-1] + CSO_flux)/(riverQ['flow'][i] + riverQ['Qadded'][i])
        else:
            riverQ['Qadded'][i] = riverQ['Qadded'][i-1]
            riverC['concentration'][i] = riverC['concentration'][i-1]*(riverQ['flow'][i-1] + riverQ['Qadded'][i-1])/(riverQ[['flow'][i] + riverQ['Qadded'][i]])
    riverC = riverC[riverC['concentration'] != 0]
    riverC['flow'] = riverQ['flow'] + riverQ['Qadded']
    EQS_exc['concentration'] = riverC['concentration'] > EQS
    EQS_exc['X'] = riverC['X']
    EQS_exc['Y'] = riverC['Y']
    EQS_exc['node ID'] = riverC['node ID']
    EQS_exc['distance'] = riverC['distance']
    EQS_exc = EQS_exc.dropna(axis = 0)
    # drop node_id and distance columns from EQS_exc
    EQS_exc = EQS_exc.drop(columns=['node ID'])
 
    # add column with the flow for the EQS_exc nodes
    EQS_exc['Q'] = riverQ['Qadded']

    # rename concentration column to EQS
    EQS_exc = EQS_exc.rename(columns={'concentration': 'EQS'})

    # add column with the concentration for the EQS_exc nodes
    EQS_exc['concentration'] = riverC['concentration']
    return EQS_exc

# %% Karen's plots :)

# plot 1
concentrations = [1.7, 17, 170, 1700]
initial_concentrations = [0.02, 0.2, 2, 10]

plt.figure(figsize=(10, 6))

for subplot_num, c in enumerate(concentrations):
    for i_c in initial_concentrations:
        output = model(riverC, riverQ, EQS_exc, CSOData, c, i_c)
        plt.subplot(2, 2, subplot_num+1)
        plt.plot(output['distance'], output['concentration'], marker='o', linestyle='-', label=f'CSO conc={c}, Initial CSO conc={i_c}')
    plt.title(f'CSO conc={i_c}')
    plt.legend()
    plt.xlabel('Distance from the lake (m)')
    plt.ylabel('Concentration of ibuprofen (mcg/m^3)')
    plt.grid(True)
plt.tight_layout()
plt.show()

# plot 2
plt.figure(figsize=(10, 6))
plt.plot(riverC['distance'], riverC['concentration'], marker='o', linestyle='-')
plt.xlabel('Distance from the lake (m)')
plt.ylabel('Concentration of ibuprofen (mcg/m^3)')
plt.title('Concentration of ibuprofen in Mølle Å stream as a function of the distance from Lyngby Lake')
plt.grid(True)
plt.show()

# plot 3
# EQS value
eqs_value = 1700  # Replace with the actual EQS value

# Plot: Concentration of Ibuprofen Exceeding EQS vs. Distance from the Lake
plt.figure(figsize=(10, 6))
plt.plot(riverC['distance'], riverC['concentration'], color='red', marker='o', linestyle='-', markersize=5, label='Concentration')
plt.axhline(eqs_value, color='blue', linestyle='--', label=f'EQS = {eqs_value} mcg/m^3')
plt.xlabel('Distance from the Lake')
plt.ylabel('Concentration of Ibuprofen (mcg/m^3)')
plt.title('Concentration of Ibuprofen Exceeding EQS vs. Distance from the Lake')
plt.grid(True)

plt.legend(loc='center right')
plt.tight_layout()
plt.show()

# CSV file
CSO_conc = 1.7*1000 # mcg/m^3
C0 = 0.02

output = model(riverC, riverQ, EQS_exc, CSOData, CSO_conc, C0)
output.drop(columns=['distance'], inplace=True)
output.to_csv('fileWithModelResults.csv')


