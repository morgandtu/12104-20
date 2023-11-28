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
flowdata['smoothed'] = flowdata.iloc[:,1].rolling(window=1320).mean()
flowdata = flowdata.iloc[1320:,:]
NH4data = NH4data.iloc[1320:,:]

NH4flux = pd.DataFrame(columns=['time','measured','model 0','model 2','model 3','model best'])
# concentration = flux/flow so flux = concentration*flow and flow = flux/concentration
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
# best guess 19/11: 6100, -1000, 200, -800, 500
a0 = 6200
a1 = -600
a2 = 200
b1 = 400
b2 = -200

# best guess 19/11: 1000, 8*60, 8/24
c12 = 1000
c22 = 8*60
c32 = 8/24

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

import objective_functions as objFun
out0 = NH4.NH4inletModel0(par0,flowdata)
NH4flux['model 0'] = out0['simNH4load']
RMSE0 = objFun.RMSE(NH4flux['measured'],NH4flux['model 0'])*100
MARE0 = objFun.MARE(NH4flux['measured'],NH4flux['model 0'])*100

out2 = NH4.NH4inletModel2(par2,flowdata)
NH4flux['model 2'] = out2['simNH4load']
RMSE2 = objFun.RMSE(NH4flux['measured'],NH4flux['model 2'])*100
MARE2 = objFun.MARE(NH4flux['measured'],NH4flux['model 2'])*100

out3 = NH4.NH4inletModel3(par3,flowdata,flowThr)
NH4flux['model 3'] = out3['simNH4load']
RMSE3 = objFun.RMSE(NH4flux['measured'],NH4flux['model 3'])*100
MARE3 = objFun.MARE(NH4flux['measured'],NH4flux['model 3'])*100

flowdatacal = flowdata.iloc[0:32335]
flowdataval = flowdata.iloc[32335:64671]
NH4fluxcal = NH4flux.iloc[0:32335]
NH4fluxval = NH4flux.iloc[32335:64671]

# flux plot
plt.figure()
# ax1=plt.subplot(2,1,1)
plt.plot_date(flowdatacal['time'], NH4fluxcal['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 0'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 2'], color = 'green', linestyle = '-', marker = "", label = 'model 2',linewidth=1)
plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 3'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
plt.ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.grid(True)
plt.title('Smoothed Flux: Measured vs. Modeled')
plt.legend()

# ax2=plt.subplot(2,1,2)
# plt.plot_date(flowdatacal['time'], NH4fluxcal['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
# plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 0'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
# plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 2'], color = 'green', linestyle = '-', marker = "", label = 'model 2',linewidth=1)
# plt.plot_date(NH4fluxcal['time'], NH4fluxcal['model 3'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
# ax2.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
# plt.xlim([datetime.date(2018, 6, 16), datetime.date(2018, 6, 20)])
# plt.ylim([3000, 17500])
# ax2.grid(True)
# plt.legend()

# %% getting sensitivities
# =============================================================================
# Task 2
# =============================================================================
# Local sensitivity analysis (Marius)

# OAT Sensitivity analysis
parSample0 = np.tile(par2,(9,1))
dPar = 0.1
for i in range(8):
    parSample0[i+1,i] = parSample0[i+1,i]*(1+dPar)

# Reference
out0 = NH4.NH4inletModel2(parSample0[0,:], flowdata)
out01 = objFun.RMSE(NH4flux['model 2'],out0['simNH4load'])+0.1

outA1 = NH4.NH4inletModel2(parSample0[1,:], flowdata)
outB1 = objFun.RMSE(NH4flux['model 2'],outA1['simNH4load'])+0.1

outA2 = NH4.NH4inletModel2(parSample0[2,:], flowdata)
outB2 = objFun.RMSE(NH4flux['model 2'],outA2['simNH4load'])+0.1

outA3 = NH4.NH4inletModel2(parSample0[3,:], flowdata)
outB3 = objFun.RMSE(NH4flux['model 2'],outA3['simNH4load'])+0.1

outA4 = NH4.NH4inletModel2(parSample0[4,:], flowdata)
outB4 = objFun.RMSE(NH4flux['model 2'],outA4['simNH4load'])+0.1

outA5 = NH4.NH4inletModel2(parSample0[5,:], flowdata)
outB5 = objFun.RMSE(NH4flux['model 2'],outA5['simNH4load'])+0.1

outA6 = NH4.NH4inletModel2(parSample0[6,:], flowdata)
outB6 = objFun.RMSE(NH4flux['model 2'],outA6['simNH4load'])+0.1

outA7 = NH4.NH4inletModel2(parSample0[7,:], flowdata)
outB7 = objFun.RMSE(NH4flux['model 2'],outA7['simNH4load'])+0.1

outA8 = NH4.NH4inletModel2(parSample0[8,:], flowdata)
outB8 = objFun.RMSE(NH4flux['model 2'],outA8['simNH4load'])+0.1

# Last Conc
# RMSE
Si_RMSE=np.zeros(8)
Si_RMSE[0]=((outB1-out01)/out01)/dPar
Si_RMSE[1]=((outB2-out01)/out01)/dPar
Si_RMSE[2]=((outB3-out01)/out01)/dPar
Si_RMSE[3]=((outB4-out01)/out01)/dPar
Si_RMSE[4]=((outB5-out01)/out01)/dPar
Si_RMSE[5]=((outB6-out01)/out01)/dPar
Si_RMSE[6]=((outB7-out01)/out01)/dPar
Si_RMSE[7]=((outB8-out01)/out01)/dPar

#  Global sensitivity analysis

param = np.array([('a0', 5000, 7000), ('a1', -2000, 2000), ('a2', -2000, 2000), ('b1', -2000, 2000), ('b2', -2000, 2000), ('c1', 400, 800), ('c2', 1*60, 12*60), ('c3', 0/24, 24/24)], dtype=[('name', 'U10'), ('min', 'f4'), ('max', 'f4')])

# regression GSA
# import GSAregression_asst4 as GSA_regression_fun
# sRegr = GSA_regression_fun.GSAregression(param,flowdata,1000)
# print(abs(sRegr))

# morris GSA
n_trj = 50
import morrisGSA as morrisGSA_fun
[sMorris,dMorris] = morrisGSA_fun.morrisGSA(n_trj,param,flowdata)
print(abs(sMorris))



# %% Global sensitivity analysis

param = np.array([('a0', 5000, 7000), ('a1', -2000, 2000), ('a2', -2000, 2000), ('b1', -2000, 2000), ('b2', -2000, 2000), ('c1', 400, 800), ('c2', 1*60, 12*60), ('c3', 0/24, 24/24)], dtype=[('name', 'U10'), ('min', 'f4'), ('max', 'f4')])

# regression GSA
# import GSAregression_asst4 as GSA_regression_fun
# sRegr = GSA_regression_fun.GSAregression(param,flowdata,1000)
# print(abs(sRegr))

# morris GSA
n_trj = 50
import morrisGSA as morrisGSA_fun
[sMorris,dMorris] = morrisGSA_fun.morrisGSA(n_trj,param,flowdata)
print(abs(sMorris))


# %% Calibration

# best fit values
a0 = 6200 #6000
a1 = -600 #-1200
a2 = 200 #800
b1 = 400 #700
b2 = -200 #-200
c1 = 1000 #1000, not optimized
c2 = 8*60 #8*60, not optimized
c3 = 8/24 #8*24

# %% Functions and all that etc--COLLAPSE THIS
# making param arrays
params = np.zeros(8)
params[0] = a0
params[1] = a1
params[2] = a2
params[3] = b1
params[4] = b2
params[5] = c1
params[6] = c2
params[7] = c3

# split dataset
flowdatacal = flowdata.iloc[0:32335]
flowdataval = flowdata.iloc[32335:64671]
NH4fluxcal = NH4flux.iloc[0:32335]
NH4fluxval = NH4flux.iloc[32335:64671]

def NH4inletModel2_fit(param, flowdata):
    import numpy as np
    import NH4models as NH4
    
    sim_out = NH4.NH4inletModel2(param, flowdata)
    
    return sim_out

fluxWrap = NH4inletModel2_fit(params, flowdatacal)

def run_NH4inletModel2_fit(params, flowdata, NH4flux):
    import objective_functions as objFun
    
    sim_out = NH4inletModel2_fit(params, flowdata)
    RMSE = objFun.RMSE(NH4flux['measured'],sim_out['simNH4load'])

    return RMSE

goodnessoffit = run_NH4inletModel2_fit(params, flowdatacal, NH4fluxcal)

def run_NH4inletModel2_fit_bnd(params, flowdata, NH4flux, lb, ub):
    import objective_functions as objFun
    if all(lb < params) & all(params < ub):
        sim_out =  NH4inletModel2_fit(params, flowdata)
        RMSE = objFun.RMSE(NH4flux['measured'],sim_out['simNH4load'])
    else:
        RMSE = 1e300
    return RMSE

param = np.array([('a0', 5700, 6200), 
                 ('a1', -1000, 1000), 
                 ('a2', -1000, 1000), 
                 ('b1', -1000, 1000),
                 ('b2', -1000, 1000),
                 ('c1', 999, 1001),
                 ('c2', 7.9*60, 8.1*60),
                 ('c3', 0, 1)],
                 dtype=[('name', 'U10'), ('min', 'f4'), ('max', 'f4')])

ParBounds = [(5500, 6500), (-2000, 2000), (-2000, 2000), (-2000, 2000), (-2000, 1000), (999, 1001), (7.9*60, 8.1*60), (0, 1)]

goodnessOfFit = run_NH4inletModel2_fit_bnd(params, flowdatacal, NH4fluxcal, param['min'], param['max'])

# %% optimization

# assign starting values
x0 = [a0,a1,a2,b1,b2,c1,c2,c3]
from scipy.optimize import minimize
from scipy.optimize import shgo

# run local optimizer
resOptimizer = minimize(run_NH4inletModel2_fit_bnd, x0, args=(flowdatacal, NH4fluxcal, param['min'], param['max']), method='nelder-mead',options={'disp': True}, tol = 1e-5)
bestParSet= resOptimizer ['x']

# run global optimizer
# resOptimizer = shgo(run_NH4inletModel2_fit_bnd, ParBounds, args=(flowdatacal, NH4fluxcal, param['min'], param['max']), n=50,iters=2,options={'disp': True})
# bestParSet= resOptimizer ['x']

# %% rerunning with best par set
bestparams = np.zeros(8)
bestparams[0] = bestParSet[0]
bestparams[1] = bestParSet[1]
bestparams[2] = bestParSet[2]
bestparams[3] = bestParSet[3]
bestparams[4] = bestParSet[4]
bestparams[5] = c1
bestparams[6] = c2
bestparams[7] = bestParSet[7]
outbest = NH4.NH4inletModel2(bestparams, flowdata)
NH4flux['model best'] = outbest['simNH4load']
# flow = flux/concentration
bestflow = outbest['simNH4load']/outbest['simNH4conc']
bestflow = bestflow.iloc[32335:64671]
RMSEoverallbest = objFun.RMSE(NH4flux['measured'],NH4flux['model best'])*100

plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot(NH4fluxval['time'], NH4fluxval['model best'], label='model best',color='green', linewidth=1)
#plt.plot(NH4flux['time'], NH4flux['model 2'], label='model 2',color='red', linewidth=1)
plt.plot(NH4fluxval['time'], NH4fluxval['measured'], label='measured',color='black', linewidth=1)
ax1.set_ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.grid(True)
plt.legend()
plt.title('Smoothed Flux: Measured vs. Modeled')
ax1=plt.subplot(2,1,2)
plt.plot(flowdataval['time'], flowdataval['smoothed'], label='measured',color='green', linewidth=1)
plt.plot(flowdataval['time'], bestflow, label='model best',color='black', linewidth=1)

NH4fluxcal = NH4flux.iloc[0:32335]
NH4fluxval = NH4flux.iloc[32335:64671]

# %% calculating error again in a different way

RMSEcalibrated = objFun.RMSE(NH4fluxcal['measured'],NH4fluxcal['model best'])*100
RMSEvalidated = objFun.RMSE(NH4fluxval['measured'],NH4fluxval['model best'])*100
NH4fluxval2 = NH4flux.iloc[40569:58115]
RMSEvalidated2 = objFun.RMSE(NH4fluxval2['measured'],NH4fluxval2['model best'])*100

# %% calculating error cal--WRONG, COLLAPSE

flowThr = 250

# define events
# you find the values of your data above the threshold
idxFlowAboveThr = flowdatacal['flow']>flowThr # this gives you a Boolean vector
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
dummySimulation = np.array([]) #the array to save the aggregate results
dummyObservation = np.array([]) #the array to save the aggregate measurements

# loop over events
for ev in range (len(idxEvStart)):
    startDate = flowdatacal['time'][idxEvStart[ev]]
    stopDate = flowdatacal['time'][idxEvEnd[ev]]
    stopDate=pd.to_datetime(stopDate) + pd.DateOffset(2) # lag time
    # find indices of the event
    idxEv=(flowdatacal['time']>startDate) & (flowdatacal['time']<=stopDate)
    # running a dummy simulation
    dummySimulation = np.concatenate((dummySimulation, NH4fluxcal['model best'][idxEv]))
    dummyObservation = np.concatenate((dummyObservation, NH4fluxcal['measured'][idxEv]))
RMSEcal = objFun.RMSE(dummyObservation, dummySimulation)*100

# %% doing it again for val data--WRONG, COLLAPSE

# wet weather event threshold (m^3/hour)
flowThr = 250

# define events
# you find the values of your data above the threshold
idxFlowAboveThr = flowdataval['flow']>flowThr # this gives you a Boolean vector
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
dummySimulation = np.array([]) #the array to save the aggregate results
dummyObservation = np.array([]) #the array to save the aggregate measurements

for ev in range (len(idxEvStart)):
    startDate = flowdataval['time'][idxEvStart[ev]]
    stopDate = flowdataval['time'][idxEvEnd[ev]]
    stopDate=pd.to_datetime(stopDate) + pd.DateOffset(2) # lag time
    # find indices of the event
    idxEv=(flowdataval['time']>startDate) & (flowdataval['time']<=stopDate)
    # running a dummy simulation
    dummySimulation = np.concatenate((dummySimulation, NH4fluxval['model best'][idxEv]))
    dummyObservation = np.concatenate((dummyObservation, NH4fluxval['measured'][idxEv]))
RMSEval = objFun.RMSE(dummyObservation, dummySimulation)*100

# %% val data without wet weather event--WRONG, COLLAPSE

flowdataval2 = flowdata.iloc[39882:57613]
NH4fluxval2 = NH4flux.iloc[39882:57613]

# wet weather event threshold (m^3/hour)
flowThr = 250

# define events
# you find the values of your data above the threshold
idxFlowAboveThr = flowdataval2['flow']>flowThr # this gives you a Boolean vector
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
dummySimulation = np.array([]) #the array to save the aggregate results
dummyObservation = np.array([]) #the array to save the aggregate measurements

for ev in range (len(idxEvStart)):
    startDate = flowdataval2['time'][idxEvStart[ev]]
    stopDate = flowdataval2['time'][idxEvEnd[ev]]
    stopDate=pd.to_datetime(stopDate) + pd.DateOffset(2) # lag time
    # find indices of the event
    idxEv=(flowdataval2['time']>startDate) & (flowdataval2['time']<=stopDate)
    # running a dummy simulation
    dummySimulation = np.concatenate((dummySimulation, NH4fluxval2['model best'][idxEv]))
    dummyObservation = np.concatenate((dummyObservation, NH4fluxval2['measured'][idxEv]))

RMSEval2 = objFun.RMSE(dummyObservation, dummySimulation)*100

# %% plot observed vs simulated

plt.figure()
plt.plot_date(NH4fluxval['time'], NH4fluxval['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
plt.plot_date(NH4fluxval['time'], NH4fluxval['model best'], color = 'red', linestyle = '-', marker = "", label = 'simulated',linewidth=1)
#plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
#plt.plot_date(flowdataval['time'], NH4fluxval['model 2'], color = 'green', linestyle = '-', marker = "", label = 'original model',linewidth=1)
#plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
plt.ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.grid(True)
plt.title('Optimized Model vs. Measured Data')
plt.legend()

plt.figure()
plt.plot_date(NH4fluxval['time'], NH4fluxval['measured'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
plt.plot_date(NH4fluxval['time'], NH4fluxval['model best'], color = 'red', linestyle = '-', marker = "", label = 'simulated',linewidth=1)
#plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
#plt.plot_date(flowdataval['time'], NH4fluxval['model 2'], color = 'green', linestyle = '-', marker = "", label = 'original model',linewidth=1)
#plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
plt.ylabel('NH$4^+$ flux $(g/hour)$') #add y-label
plt.grid(True)
plt.xlim([datetime.date(2018, 7, 25), datetime.date(2018, 7, 28)])
plt.ylim([4000,8000])
plt.title('Optimized Model vs. Measured Data')
plt.legend()

# concentration
# plt.figure()
# plt.plot_date(NH4dataval['time'], NH4dataval['nh4'], color = 'black', linestyle = '-', marker = "", label = 'measured',linewidth=1)
# plt.plot_date(outbest['time'], outbest['simNH4conc'], color = 'red', linestyle = '-', marker = "", label = 'simulated',linewidth=1)
#plt.plot_date(out0['time'], out0['simNH4load'], color = 'red', linestyle = '-', marker = "", label = 'model 0',linewidth=1)
#plt.plot_date(flowdataval['time'], NH4fluxval['model 2'], color = 'green', linestyle = '-', marker = "", label = 'original model',linewidth=1)
#plt.plot_date(out3['time'], out3['simNH4load'], color = 'blue', linestyle = '-', marker = "", label = 'model 3',linewidth=1)
# plt.ylabel('NH$4^+$ concentration $(g/m^3)$') #add y-label
# plt.grid(True)
# plt.title('Optimized Model vs. Measured Data')
# plt.legend()


