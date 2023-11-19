# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:43:38 2020

@author: luve
Models for predicting NH4 concentrations - assignment 4 in 12104

"""

import numpy as np
import pandas as pd

def NH4inletModel0(param,inputData):
    
    
# ============================================================================
# MODEL #0 - Basic NH4 inlet model, based on the Fourier series
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#
#        - inputData: dataframe with flow data 

# ============================================================================

    # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    # calculate concentration
    Q=inputData['flow'] #[m3/hr]
    NH4conc=DWFload/Q   #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel1(param,inputData):
    
    
# ============================================================================
# MODEL #1 - Extension of model0 with tank in series to account for volume placed
#            before sensors (e.g. primary clarifies)
# Inputs - Pars: array containing 
#              - the five parameters of the Fourier series according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#              - V = volume placed before the sensor
#        - inputData: dataframe with flow data 

# ============================================================================
    # reset index of the input dataframe, since Python weirdly keeps the old numbering
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
    # get the timestep of the time series
    dt=np.median(timeVector.to_series().diff()) # calculate the median difference between values (in case some data are missing)
    dt=dt/np.timedelta64(1, 'h')  # convert to hours (as float)
    
    # get flow data
    Q=inputData['flow']

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    # ============  the upstream buffer volume ===============================
    noTanks=3 # number of conceptual CTSR used to represent advection in ups volume
    
    V=param[5]/noTanks # volume of each single conceputal tank
    noTimeStep=inputData.shape[0]
    M=np.empty([noTimeStep,noTanks]) #inizialize matrix with mass of each tank [g]
    DWFload_OutTank=np.empty([noTimeStep]) #inizialize matrix with mass of each tank [g]
    C0=DWFload[0]/Q[0] # [g/m3] assume an initial concentration, based on the first simulated value
    
    # use this concentration to initialize tanks
    for i in range(0,(noTanks)):
        M[0,i]=C0*V #[g]
    
    # Solve mass balance for the three tanks for all the simulation steps
    # general mass balance M[t]=M[t-1]+inFlux[t-1]-outFlux[t-1]

    for i in (range(1,noTimeStep)):
        # mass balance Tank #0 
        M[i,0]=M[i-1,0]+(DWFload[i-1] -        Q[i-1]*(M[i-1,0]/V))*dt #[g]
        # mass balance Tank #1
        M[i,1]=M[i-1,1]+(Q[i-1]*(M[i-1,0]/V) - Q[i-1]*(M[i-1,1]/V))*dt #[g]
        # mass balance Tank #2
        M[i,2]=M[i-1,2]+(Q[i-1]*(M[i-1,1]/V) - Q[i-1]*(M[i-1,2]/V))*dt #[g]
        # flux out of tanks
        DWFload_OutTank[i]=(M[i,noTanks-1]/V)*Q[i-1] #[g/hr]
    
    
    # calculate concentration
    NH4conc=DWFload_OutTank/Q  #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_OutTank
    out['simNH4conc']=NH4conc

    return out

def NH4inletModel2(param,inputData):
    
# ============================================================================
# MODEL #2 - Extension of model0 with an additional spike to describe steep 
#            rising in NH4 loads (tipically in the morning)
# Inputs - Pars: array containing 
#              - the eight parameters (five of of the Fourier series + three of the pulse)
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t) + 
#                           c1 * exp(-c2*(log10(t)-log10(c3))^2)
#
#
#        - inputData: dataframe with flow data 

# ============================================================================
    # reset index of the input dataframe, since Python weirdly keeps the old numbering
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
   
    # get flow data
    Q=inputData['flow']

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) + 
             param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY) +                  
             param[5]*np.exp(-param[6]*(np.log10(timeDOY)-np.log10(param[7]))**2))
    
    # calculate concentration
    NH4conc=DWFload/Q  #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel3(param,inputData,flowThr):
    
# ============================================================================
# MODEL #3 - Extension of model0 with a dilution model for when the event is ending 
#           similar to the restoration phase in Langeveld et al. (2017)
#            
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#               plus three additional parameters for defining the dilution (dF) in the restoration phase 
#               dF(tSEE)=c1*exp(-((log(tSEE)-c2)**2 )/c3))
#               with tSEE=time Since End of Event
#        - inputData: dataframe with flow data 
#        - flowThr: threshold for defining start and end of rain event
#
# ============================================================================
   # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    
    # =================  identify end of wet event =======================
    Q=inputData['flow'] #[m3/hr]    
    idxFlowAboveThr=Q>flowThr #identify values when flow was above flow threshold
    
    # calculate the time passed since the last event
    timeStepSinceLastEvent=np.zeros(len(inputData))
    timeStepSinceLastEvent[0]=1440*365 #assume the start of the model in dry conditions (just put an unrealistic big value, such as 1 year)
    for i in range(1,len(inputData)): # loop over events
        if not idxFlowAboveThr[i]:
            timeStepSinceLastEvent[i]=timeStepSinceLastEvent[i-1]+1
            

    np.seterr(divide = 'ignore') #change settings - ignore error for log of zero
    dilutionFactor=(1-param[5]*np.exp(-((np.log(timeStepSinceLastEvent)-param[6])**2 )/param[7]))
    np.seterr(divide = 'warn') # re-activate error
    
    DWFload_new=DWFload*dilutionFactor
    
    
    # calculate concentration
    NH4conc=DWFload_new/Q   #[g/m3]
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_new
    out['simNH4conc']=NH4conc

    return out


def NH4inletModel4(param,inputData,flowThr):
    
# ============================================================================
# MODEL #4 - combination of Model 1 (tanks in series) annd model 3 (dilution)
#            
# Inputs - Pars: array containing the five parameters of the Fourier series
#                according to the formulation
#                F(t)= a0 + a1*sin(2*pi*t) + b1 * cos(2*pi*t) +
#                           a2*sin(4*pi*t) + b2 * cos(4*pi*t)
#               plus three additional parameters for defining the dilution (dF) in the restoration phase 
#               dF(tSEE)=c1*exp(-((log(tSEE)-c2)**2 )/c3))
#               with tSEE=time Since End of Event
#               and V = volume placed before the sensor
#        - inputData: dataframe with flow data 
#        - flowThr: threshold for defining start and end of rain event
#
# ============================================================================
   # reset index of the input dataframe, since Python weirdly keeps the old numbering    
    inputData=inputData.reset_index(drop=True)

    ## convert time stamps into fraction of a day
    # extract time column into DateTimeIndex vector
    timeVector=pd.DatetimeIndex(inputData['time'])
                             
    # recalculate time vector as fraction of a day
    timeDOY = timeVector.hour/24+timeVector.minute/1440
    
    # get the timestep of the time series
    dt=np.median(timeVector.to_series().diff()) # calculate the median difference between values (in case some data are missing)
    dt=dt/np.timedelta64(1, 'h')  # convert to hours (as float)
    

    # ================= the basic DWF model =================
    # calculate DWF profile for ammonia [g/hr]
    DWFload=(param[0]+param[1]*np.sin(2*np.pi*timeDOY) + param[2]*np.cos(2*np.pi*timeDOY) +
                      param[3]*np.sin(4*np.pi*timeDOY) + param[4]*np.cos(4*np.pi*timeDOY))
    
    
    # =================  identify end of wet event =======================
    Q=inputData['flow'] #[m3/hr]    
    idxFlowAboveThr=Q>flowThr #identify values when flow was above flow threshold
    
    # calculate the time passed since the last event
    timeStepSinceLastEvent=np.zeros(len(inputData))
    timeStepSinceLastEvent[0]=1440*365 #assume the start of the model in dry conditions (just put an unrealistic big value, such as 1 year)
    for i in range(1,len(inputData)): # loop over events
        if not idxFlowAboveThr[i]:
            timeStepSinceLastEvent[i]=timeStepSinceLastEvent[i-1]+1
            
    np.seterr(divide = 'ignore') #change settings - ignore error for log of zero
    dilutionFactor=(1-param[5]*np.exp(-((np.log(timeStepSinceLastEvent)-param[6])**2 )/param[7]))
    np.seterr(divide = 'warn') # re-activate error
    
    DWFload_new=DWFload*dilutionFactor
    
   # ============  the upstream buffer volume ===============================
    noTanks=3 # number of conceptual CTSR used to represent advection in ups volume
    
    V=param[8]/noTanks # volume of each single conceputal tank
    noTimeStep=inputData.shape[0]
    M=np.empty([noTimeStep,noTanks]) #inizialize matrix with mass of each tank [g]
    DWFload_OutTank=np.empty([noTimeStep]) #inizialize matrix with mass of each tank [g]
    C0=DWFload_new[0]/Q[0] # [g/m3] assume an initial concentration, based on the first simulated value
    
    # use this concentration to initialize tanks
    for i in range(0,(noTanks)):
        M[0,i]=C0*V #[g]
    
    # Solve mass balance for the three tanks for all the simulation steps
    # general mass balance M[t]=M[t-1]+inFlux[t-1]-outFlux[t-1]

    for i in (range(1,noTimeStep)):
        # mass balance Tank #0 
        M[i,0]=M[i-1,0]+(DWFload_new[i-1] -        Q[i-1]*(M[i-1,0]/V))*dt #[g]
        # mass balance Tank #1
        M[i,1]=M[i-1,1]+(Q[i-1]*(M[i-1,0]/V) - Q[i-1]*(M[i-1,1]/V))*dt #[g]
        # mass balance Tank #2
        M[i,2]=M[i-1,2]+(Q[i-1]*(M[i-1,1]/V) - Q[i-1]*(M[i-1,2]/V))*dt #[g]
        # flux out of tanks
        DWFload_OutTank[i]=(M[i,noTanks-1]/V)*Q[i-1] #[g/hr]
    
    
    # calculate concentration
    NH4conc=DWFload_OutTank/Q  #[g/m3]    
    
    
    # prepare output of the function
    out=pd.DataFrame(inputData['time'])
    out['simNH4load']=DWFload_OutTank
    out['simNH4conc']=NH4conc

    return out
