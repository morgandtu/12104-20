#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:28:29 2023

@author: benp.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:26:53 2023

@author: benp.
"""

# ASSIGNMENT 4 WOOOOO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from IPython import get_ipython
import datetime
import os
interactive(True)
plt.close("all") # close all figures
plt.rcParams.update({'font.size': 9}) # set font size in the graph
'/Users/benp./Desktop/12104 Modeling/Module 4/Assignment4.py'
os.chdir('/Users/benp./Desktop/12104 Modeling/Module 4')


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

NH4flux = pd.DataFrame(columns=['time','measured','model 0','model 2','model 3'])
# concentration = flux/flow so flux = concentration*flow
NH4flux['measured'] = (NH4data['nh4'])*flowdata['smoothed'] # m^3/hr * g/m^3 = g/hr
NH4flux['time'] = flowdata['time']

# plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
# plt.plot_date(flowdata['time'], flowdata['smoothed'], color = 'red', linestyle = '--', marker = "", label = 'smoothed')
# plt.title('Flow and NH$4^+$ in Viby, 13/06/2018-01/08/2018')
# plt.ylim([50, 500])
# plt.legend()
# plt.grid()

# %% subplot for 1.1

plt.figure()
# flow
ax1=plt.subplot(4,1,1)
plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
plt.plot_date(flowdata['time'], flowdata['smoothed'], color = 'red', linestyle = '--', marker = "", label = 'smoothed')
ax1.set_ylabel('Flow $(m^3/hour)$') #add y-label
plt.title('Flow and NH$4^+$ in Viby, 13/06/2018-01/08/2018')
plt.legend()
# concentration
ax2=plt.subplot(4,1,2)
plt.plot_date(flowdata['time'], NH4data['nh4'], color = 'red', linestyle = '-', marker = "", label = 'measured')
ax2.set_ylabel('NH$4^+$ Concentration $(g/m^3)$') #add y-label
# flux
ax3=plt.subplot(4,1,3)
plt.plot_date(flowdata['time'], NH4flux['measured'], color = 'green', linestyle = '-', marker = "", label = 'measured')
ax3.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
# day
ax4=plt.subplot(4,1,4)
plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "")
ax4.set_ylabel('Flow $(m^3/hour)$') #add y-label
plt.xlim([datetime.date(2018, 6, 23), datetime.date(2018, 6, 24)])
plt.ylim([25, 250])
# making it look nice
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)


# plt.figure()
# plt.plot_date(flowdata['time'], flowdata['flow'], color = 'blue', linestyle = '-', marker = "", label = 'flow')
# plt.ylabel('Flow $(m^3/hour)$') #add y-label
# plt.title('A Day of Flow in Viby')
# # xt=pd.date_range(start = min(flowdata['time']), end = max(flowdata['time']),freq="6H")
# # plt.xticks(xt)
# # plt.xticklabels(xt.strftime("%Y-%m-%d"))
# import datetime
# plt.xlim([datetime.date(2018, 6, 23), datetime.date(2018, 6, 25)])
# plt.ylim([25, 250])
# plt.grid(True) #add grid to graphs
# plt.legend()

# %% model testing

import NH4models as NH4
# parameters--medians from Pedersen
# best guess 19/11: 6100, -1500, 200, -800, 50
a0 = 6000
a1 = -1500
a2 = 200
b1 = -800
b2 = 50

# best guess 19/11: 600, 4*60, 11/24
c12 = 600
c22 = 4*60
c32 = 11/24

# best guess 0.2, 3, 4, 250
c13 = 0.2
c23 = 3
c33 = 4
flowThr = 250

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
plt.plot_date(flowdata['time'], NH4flux['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
plt.plot_date(out2['time'], out2['simNH4load'], color = 'green', linestyle = '-', marker = "", label = 'model 2',linewidth=1)
plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
ax1.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
ax1.grid(True)
plt.title('Smoothed Flux: Measured vs. Modeled')
plt.legend()

ax2=plt.subplot(2,1,2)
plt.plot_date(flowdata['time'], NH4flux['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
plt.plot_date(out2['time'], out2['simNH4load'], color = 'green', linestyle = '-', marker = "", label = 'model 2',linewidth=1)
plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
ax2.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.xlim([datetime.date(2018, 6, 16), datetime.date(2018, 6, 20)])
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


# %% objective function analysis

# wet weather event threshold (m^3/hour)
flowThr = 250

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

# declare vectors for both simulations
dummySimulation0 = np.array([]) #the array to save the aggregate results
dummyObservation0 = np.array([]) #the array to save the aggregate measurements
dummySimulation2 = np.array([]) #the array to save the aggregate results
dummyObservation2 = np.array([]) #the array to save the aggregate measurements
dummySimulation3 = np.array([]) #the array to save the aggregate results
dummyObservation3 = np.array([]) #the array to save the aggregate measurements

NH4flux['model 0'] = out0['simNH4load']
NH4flux['model 2'] = out2['simNH4load']
NH4flux['model 3'] = out3['simNH4load']

import objective_functions as objFun
# loop over events
for ev in range (len(idxEvStart)):
    startDate = flowdata['time'][idxEvStart[ev]]
    stopDate = flowdata['time'][idxEvEnd[ev]]
    stopDate=pd.to_datetime(stopDate) + pd.DateOffset(2) # lag time
    # find indices of the event
    idxEv=(flowdata['time']>startDate) & (flowdata['time']<=stopDate)
    # running a dummy simulation
    dummySimulation0 = np.concatenate((dummySimulation0, NH4flux['model 0'][idxEv]))
    dummyObservation0 = np.concatenate((dummyObservation0, NH4flux['measured'][idxEv]))
    dummySimulation2 = np.concatenate((dummySimulation2, NH4flux['model 2'][idxEv]))
    dummyObservation2 = np.concatenate((dummyObservation2, NH4flux['measured'][idxEv]))
    dummySimulation3 = np.concatenate((dummySimulation3, NH4flux['model 3'][idxEv]))
    dummyObservation3 = np.concatenate((dummyObservation3, NH4flux['measured'][idxEv]))

MARE0 = objFun.MARE(dummyObservation0, dummySimulation0)*100
MARE2 = objFun.MARE(dummyObservation2, dummySimulation2)*100
MARE3 = objFun.MARE(dummyObservation3, dummySimulation3)*100

# %% 
# =============================================================================
# Task 2
# =============================================================================
# Local sensitivity analysis (Marius)

import objective_functions as objFun

# OAT Sensitivity analysis
parSample0 = np.tile(par2,(9,1))
dPar = 0.1
for i in range(8):
    parSample0[i+1,i] = parSample0[i+1,i]*(1+dPar)

# Reference
out0 = NH4.NH4inletModel2(parSample0[0,:], flowdata)
out01 = objFun.MARE(NH4flux['model 2'],out0['simNH4load'])+0.1

outA1 = NH4.NH4inletModel2(parSample0[1,:], flowdata)
outB1 = objFun.MARE(NH4flux['model 2'],outA1['simNH4load'])+0.1

outA2 = NH4.NH4inletModel2(parSample0[2,:], flowdata)
outB2 = objFun.MARE(NH4flux['model 2'],outA2['simNH4load'])+0.1

outA3 = NH4.NH4inletModel2(parSample0[3,:], flowdata)
outB3 = objFun.MARE(NH4flux['model 2'],outA3['simNH4load'])+0.1

outA4 = NH4.NH4inletModel2(parSample0[4,:], flowdata)
outB4 = objFun.MARE(NH4flux['model 2'],outA4['simNH4load'])+0.1

outA5 = NH4.NH4inletModel2(parSample0[5,:], flowdata)
outB5 = objFun.MARE(NH4flux['model 2'],outA5['simNH4load'])+0.1

outA6 = NH4.NH4inletModel2(parSample0[6,:], flowdata)
outB6 = objFun.MARE(NH4flux['model 2'],outA6['simNH4load'])+0.1

outA7 = NH4.NH4inletModel2(parSample0[7,:], flowdata)
outB7 = objFun.MARE(NH4flux['model 2'],outA7['simNH4load'])+0.1

outA8 = NH4.NH4inletModel2(parSample0[8,:], flowdata)
outB8 = objFun.MARE(NH4flux['model 2'],outA8['simNH4load'])+0.1

# Last Conc
# MARE
Si_MARE=np.zeros(8)
Si_MARE[0]=((outB1-out01)/out01)/dPar
Si_MARE[1]=((outB2-out01)/out01)/dPar
Si_MARE[2]=((outB3-out01)/out01)/dPar
Si_MARE[3]=((outB4-out01)/out01)/dPar
Si_MARE[4]=((outB5-out01)/out01)/dPar
Si_MARE[5]=((outB6-out01)/out01)/dPar
Si_MARE[6]=((outB7-out01)/out01)/dPar
Si_MARE[7]=((outB8-out01)/out01)/dPar


# %% Global sensitivity analysis

param = np.array([('a0', 1000, 7000), ('a1', -2000, 2000), ('a2', -2000, 2000), ('b1', -2000, 2000), ('b2', -2000, 2000), ('c1', 400, 800), ('c2', 1*60, 12*60), ('c3', 0/24, 24/24)], dtype=[('name', 'U10'), ('min', 'f4'), ('max', 'f4')])

# regression GSA
# import GSAregression_asst4 as GSA_regression_fun
# sRegr = GSA_regression_fun.GSAregression(param,flowdata,1000)
# print(abs(sRegr))

# morris GSA
n_trj = 50
import morrisGSA as morrisGSA_fun
[sMorris,dMorris] = morrisGSA_fun.morrisGSA(n_trj,param,flowdata)
print(abs(sMorris))


