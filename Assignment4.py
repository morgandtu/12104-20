#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:26:53 2023

@author: benp.
"""

# ASSIGNMENT 4 WOOOOO

import os #optional library (to define working directory and avoid issues)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from IPython import get_ipython
interactive(True)
plt.close("all") # close all figures
plt.rcParams.update({'font.size': 9}) # set font size in the graph

# %% ==========================================================================
# Task 1
# =============================================================================

flowdata = pd.read_csv('dataViby_flow_180613_180801.csv', sep=";", parse_dates=['time'])

NH4data = pd.read_csv('dataViby_NH4_180613_180801.csv', sep=";", parse_dates=['time']) # mg/L aka g/m^3
NH4data['nh4'].iloc[21677] = (NH4data['nh4'].iloc[21676] + NH4data['nh4'].iloc[21678]) / 2
NH4data['nh4'].iloc[33000] = (NH4data['nh4'].iloc[32999] + NH4data['nh4'].iloc[33001]) / 2
NH4data['nh4'].iloc[34547] = (NH4data['nh4'].iloc[34546] + NH4data['nh4'].iloc[34548]) / 2

# smoothing and removing NaNs
flowdata['smoothed'] = flowdata.iloc[:,1].rolling(window=720).mean()
flowdata = flowdata.iloc[719:,:]
NH4data = NH4data.iloc[719:,:]

# concentration = flux/flow so flux = concentration*flow
NH4flux = (NH4data['nh4'])*flowdata['smoothed'] # m^3/hr * g/m^3 = g/hr

# plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
# plt.plot_date(flowdata['time'], flowdata['smoothed'], color = 'red', linestyle = '--', marker = "", label = 'smoothed')
# plt.title('Flow and NH$4^+$ in Viby, 13/06/2018-01/08/2018')
# plt.ylim([50, 500])
# plt.legend()
# plt.grid()

# %% subplot for 1.1

plt.figure()
# flow
ax1=plt.subplot(3,1,1)
plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
plt.plot_date(flowdata['time'], flowdata['smoothed'], color = 'red', linestyle = '--', marker = "", label = 'smoothed')
ax1.set_ylabel('Flow $(m^3/hour)$') #add y-label
plt.title('Flow and NH$4^+$ in Viby, 13/06/2018-01/08/2018')
plt.legend()
# concentration
ax2=plt.subplot(3,1,2)
plt.plot_date(flowdata['time'], NH4data['nh4'], color = 'red', linestyle = '-', marker = "", label = 'measured')
ax2.set_ylabel('NH$4^+$ Concentration $(g/m^3)$') #add y-label
plt.legend()
# flux
ax3=plt.subplot(3,1,3)
plt.plot_date(flowdata['time'], NH4flux, color = 'green', linestyle = '-', marker = "", label = 'flux')
ax3.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.legend()
# making it look nice
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)


# %% single day plot for 1.1
plt.figure()
plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
plt.ylabel('Flow $(m^3/hour)$') #add y-label
plt.title('A Day of Flow in Viby')
# xt=pd.date_range(start = min(flowdata['time']), end = max(flowdata['time']),freq="6H")
# plt.xticks(xt)
# plt.xticklabels(xt.strftime("%Y-%m-%d"))
import datetime
plt.xlim([datetime.date(2018, 6, 23), datetime.date(2018, 6, 24)])
plt.ylim([25, 250])
plt.grid(True) #add grid to graphs
plt.legend()

# %% model testing

import NH4models as NH4
# parameters--medians from Pedersen
a0 = 6000
a1 = 500
a2 = 1000
b1 = 500
b2 = 0

c12 = 600
c22 = 5*60
c32 = 9/24

c13 = 0.2
c23 = 4
c33 = 0.5
flowThr = 300

par0 = np.zeros(8)
par0[0] = a0
par0[1] = a1
par0[2] = a2
par0[3] = b1
par0[4] = b2

par2 = np.zeros(8)
par2[0] = a0
par2[1] = a1
par2[2] = a2
par2[3] = b1
par2[4] = b2
par2[5] = c12
par2[6] = c22
par2[7] = c32

par3 = np.zeros(8)
par3[0] = a0
par3[1] = a1
par3[2] = a2
par3[3] = b1
par3[4] = b2
par3[5] = c13
par3[6] = c23
par3[7] = c33

out0 = NH4.NH4inletModel0(par0,flowdata)
out2 = NH4.NH4inletModel2(par2,flowdata)
out3 = NH4.NH4inletModel3(par3,flowdata,flowThr)

# flux plot
plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot_date(flowdata['time'], NH4flux, color = 'black', linestyle = '-', marker = "", label = 'measured')
plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0')
plt.plot_date(out2['time'], out2['simNH4load'], color = 'green', linestyle = '-', marker = "", label = 'model 2')
plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3')
ax1.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
ax1.grid(True)
plt.title('Smoothed Flux: Measured vs. Modeled')
plt.legend()

ax2=plt.subplot(2,1,2)
plt.plot_date(flowdata['time'], NH4flux, color = 'black', linestyle = '-', marker = "", label = 'measured')
plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0')
plt.plot_date(out2['time'], out2['simNH4load'], color = 'green', linestyle = '-', marker = "", label = 'model 2')
plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3')
ax2.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.xlim([datetime.date(2018, 6, 16), datetime.date(2018, 6, 22)])
plt.ylim([3000, 20000])
ax2.grid(True)
plt.legend()


# concentration plot
# plt.figure()
# ax1=plt.subplot(2,1,1)
# plt.plot_date(flowdata['time'], NH4data['nh4'], color = 'black', linestyle = '-', marker = "", label = 'measured')
# plt.plot_date(out0['time'], out0['simNH4conc'], color = 'red', linestyle = '-', marker = "", label = 'model 0')
# plt.plot_date(out2['time'], out2['simNH4conc'], color = 'green', linestyle = '-', marker = "", label = 'model 2')
# plt.plot_date(out3['time'], out3['simNH4conc'], color = 'blue', linestyle = '-', marker = "", label = 'model 3')
# ax1.set_ylabel('NH$4^+$ concentration $(g/m^3)$') #add y-label
# ax1.grid(True)
# plt.title('Concentrations: Measured vs. Modeled')
# plt.legend()

# ax2=plt.subplot(2,1,2)
# plt.plot_date(flowdata['time'], NH4data['nh4'], color = 'black', linestyle = '-', marker = "", label = 'measured')
# plt.plot_date(out0['time'], out0['simNH4conc'], color = 'red', linestyle = '-', marker = "", label = 'model 0')
# plt.plot_date(out2['time'], out2['simNH4conc'], color = 'green', linestyle = '-', marker = "", label = 'model 2')
# plt.plot_date(out3['time'], out3['simNH4conc'], color = 'blue', linestyle = '-', marker = "", label = 'model 3')
# ax2.set_ylabel('NH$4^+$ concentration $(g/m^3)$') #add y-label
# plt.xlim([datetime.date(2018, 6, 16), datetime.date(2018, 6, 22)])
# plt.ylim([0, 250])
# ax2.grid(True)
# plt.legend()


# %% objective B analysis for 1.2

# wet weather event threshold (m^3/hour)
flowThr = 300

# define events
# you find the values of your data above the threshold
idxFlowAboveThr = flowdata['flow']>flowThr # this gives you a Boolean vector
# you need to covert the Boolean (true/false) into an integer (1/0)
idxFlowAboveThr = idxFlowAboveThr.astype(int)
# now calculate where there is a change in the status (from below to above the threshold or viceversa by calculating the difference between each step
idxFlowAboveThr = idxFlowAboveThr.diff()
# find when the flow went above the threshold (from 0 to 1)
evStart = idxFlowAboveThr[idxFlowAboveThr>0]
# find when the flow went below the threshold (from 1 to 0)
evEnd = idxFlowAboveThr[idxFlowAboveThr<0]-1
# find the indices of the event start and end
idxEvStart = evStart.index
idxEvEnd = evEnd.index-1 # the step before was the last one with flow above thresh

# declare vectors
dummySimulation = np.array([]) #the array to save the aggregate results
dummyObservation = np.array([]) #the array to save the aggregate measurements

import objective_functions as objFun
# loop over events
for ev in range (len(idxEvStart)):
    startDate = flowdata['time'][idxEvStart[ev]]
    stopDate = flowdata['time'][idxEvEnd[ev]]
    stopDate=pd.to_datetime(stopDate) + pd.DateOffset(2) # lag time
    # find indices of the event
    idxEv=(flowdata['time']>startDate) & (flowdata['time']<=stopDate)
    # running a dummy simulation
    # dummySimulation = np.concatenate((dummySimulation, NH4flux['simulated'][idxEv]))
    # dummyObservation = np.concatenate((dummyObservation, NH4flux['measured'][idxEv]))

# RMSE = objFun.RMSE(dummyObservation, dummySimulation)




