# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/mariu/Documents/Python Scripts/Environmental modelling/Assignment 1')
pd.options.mode.chained_assignment = None

# housekeeping
#get_ipython().magic('reset -sf')
plt.close('all')

# getting data
riverNodes = pd.read_csv('riverNodes.csv', sep=",", decimal = ".")
distances = pd.read_csv('distances.csv', sep=",", decimal=".")

#renaming danish words
new_column_name = 'Vandmaengd'
distances = distances.rename(columns={distances.columns[12]: new_column_name})
new_column_name2 = 'Bygvaerkst'
distances = distances.rename(columns={distances.columns[7]: new_column_name2})
# removing unnecessary data
riverNodes.rename(columns={"vandfoering": "water flow (m^3/s)"}, inplace=True)
distances.rename(columns={"BI-5 (kg)": "Biological oxygen demand, 5 days (kg)"}, inplace=True)

# %%
#riverNodes.head(5)

# %%
riverNodes['beregningspunktlokalid'].unique()

# %%
distances.head(5)

# %%
# Filter rows where 'registreringfra' contains either 'Novana_Model_MOELLEAA_DK1_13500' or 'Novana_Model_MOELLEAA_DK1_3687'
idxMoelleAA = riverNodes['beregningspunktlokalid'].str.contains('MOELLEAA')

# %%
riverNodes = riverNodes[idxMoelleAA]
riverNodes = riverNodes.reset_index()
riverNodesYY = riverNodes[riverNodes["aar"]==2011]
riverNodesMM = riverNodesYY[riverNodes["maaned"]=="januar"]
idxUp= riverNodesMM['beregningspunktlokalid'].str.contains('3687')
idxDown= riverNodesMM['beregningspunktlokalid'].str.contains('13500')
#finding the index which corresponds to the inlet
idxUp_num = idxUp.where(idxUp == True).dropna().index
#finding the index which corresponds to the outlet
idxDown_num = idxDown.where(idxDown == True).dropna().index
#trasforming Int64Index type index to an integer to create an array from the index difference
idxUp_num = idxUp_num[0]
idxDown_num = idxDown_num[0]
indexdiff= abs(idxUp_num - idxDown_num)


# %%
idxUpDown = idxUp

# %%
for i in range(idxUp_num, idxDown_num+1):
    idxUpDown[i] = True

# %%
riverQ = riverNodesMM[idxUpDown]

# %%
riverQ = riverQ[['water flow (m^3/s)', 'beregningspunktlokalid', 'X', 'Y']]

# %%
riverQ.reset_index()
riverQ['distance'] = np.NaN
riverQ['CSOflow'] = np.NaN

# %%
riverC = riverQ[['beregningspunktlokalid', 'X', 'Y', 'distance']]
riverC = riverC.reset_index(drop=True)
EQS_exc= riverC.copy()

# %%
StringName=riverQ['beregningspunktlokalid'].str.split('_').str[-1]
StringName = StringName.astype(float)

# %%
type(StringName)

# %%
riverQ["distance"]= StringName - 3687
riverQ = riverQ.reset_index(drop=True)

# %%
riverC['concentration'] = 0

# %%
# starting the simple advection-dilution model 
riverC['concentration'][0] = 0.2*(10**(-6)) # initial concentration in the stream
CSO_conc = 1.7*(10**(-6)) # (g/L) assign the MP concentration in the CSO water
theta = 2.31 # %
t_CSO = 4.3 # hr


# %% 
CSOname=pd.DataFrame(np.zeros([len(riverQ),1],dtype=float),columns=['name'])

# # looping from 1 to len-1 instead of 2 to len (appendix)
# for i in range(1,len(riverQ)-1):
#     csoBool = distances['HubName'].eq(riverQ["beregningspunktlokalid"].iloc[i]) # comparing each value
#     isThereCSO = csoBool.sum() == 1 # check if there is a CSO

# %%

CSOname=pd.DataFrame(np.zeros([len(riverQ),1],dtype=float),columns=['name'])
for i in range(1,len(riverQ)-1):
    csoBool = distances['HubName'].eq(riverQ["beregningspunktlokalid"].iloc[i])
    isThereCSO = csoBool.sum() == 1 # check if there is a CSO
    if isThereCSO: 
        CSOname['name'][i] = riverQ["beregningspunktlokalid"].iloc[i]
    idxCSO = (CSOname[CSOname['name'] != 0]).index
    namesCSO = CSOname[CSOname['name'] != 0]

# Create an empty list to store the indices where the string appears
indices_where_appears = []

# Iterate through each string in namesCSO and check if it appears in distances['HubName']
for cso_name in namesCSO['name']:
    # Use str.contains to create a boolean mask
    mask = distances['HubName'].str.contains(cso_name)
    
    # Use the mask to get the indices where the string appears
    matching_indices = distances[mask].index.tolist()
    
    # Add the matching indices to the list
    indices_where_appears.extend(matching_indices)

# The list indices_where_appears now contains the indices in distances where the strings from namesCSO appear.

    
    if len(idxCSO) > 0:
        CSO_flux = 0 #create an empty value for the MP flux from the CSO
        CSO_Qtot = 0
        for j in range(0,len(idxCSO)-2): #loop over the connected CSOs
            V_CSO = distances["Vandmaengd"][idxCSO[j]]*theta # calculate the volume of the single discharge event
            Q_CSO = V_CSO/t_CSO # calculate the flow of the single discharge event
            CSO_flux = CSO_flux+Q_CSO*CSO_conc
            CSO_Qtot = CSO_Qtot+Q_CSO
        riverQ['CSOflow'][i] = riverQ['CSOflow'][i-1]+CSO_Qtot
        # calculate concentration for the node
        riverC['concentration'][i] = (riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
        riverQ['CSOflow'][i-1])+ CSO_flux)/ (riverQ['water flow (m^3/s)'][i]+
        riverQ['CSOflow'][i])
    else: # there are no CSO connected to the node
        # calculate flow added to the river by the CSO (no addition)
        riverQ['CSOflow'][i]= riverQ['CSOflow'][i-1]
        # calculate concentration for the node
        riverC['concentration'][i] =riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
        riverQ['CSOflow'][i-1])/ (riverQ['water flow (m^3/s)'][i]+
        riverQ['CSOflow'][i])
                                  
# EQS_exc= riverC["conc"]>EQS_exc


# %%

# CSOname=pd.DataFrame(np.zeros([len(riverQ),1],dtype=float),columns=['name'])

# for i in range(2,len(riverQ)):
#     csoBool = riverQ['beregningspunktlokalid'].eq(distances['HubName'].iloc[i])

#     isThereCSO = csoBool.sum() == 1 # check if there is a CSO
#     if isThereCSO: 
#         CSOname['name'][i] = riverQ["beregningspunktlokalid"].iloc[i]
#     idxCSO = (CSOname[CSOname['name'] != 0]).index
#     if len(idxCSO) > 0:
#         CSO_flux = 0 #create an empty value for the MP flux from the CSO
#         CSO_Qtot = 0
#         for j in range(0,len(idxCSO)-2): #loop over the connected CSOs
#             print(idxCSO[j])
#             V_CSO = distances["Vandmaengd"][idxCSO[j]]*theta # calculate the volume of the single discharge event
#             Q_CSO = V_CSO/t_CSO # calculate the flow of the single discharge event
#             CSO_flux = CSO_flux+Q_CSO*CSO_conc
#             CSO_Qtot = CSO_Qtot+Q_CSO
#         riverQ['CSOflow'][i] = riverQ['CSOflow'][i-1]+CSO_Qtot
#         # calculate concentration for the node
#         riverC['concentration'][i] = (riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
#         riverQ['CSOflow'][i-1])+ CSO_flux)/ (riverQ['water flow (m^3/s)'][i]+
#         riverQ['CSOflow'][i])
#     else: # there are no CSO connected to the node
#         # calculate flow added to the river by the CSO (no addition)
#         riverQ['CSOflow'][i]= riverQ['CSOflow'][i-1]
#         # calculate concentration for the node
#         riverC['concentration'][i] =riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
#         riverQ['CSOflow'][i-1])/ (riverQ['water flow (m^3/s)'][i]+
#         riverQ['CSOflow'][i])
                                  
# EQS_exc= riverC["conc"]>EQS_exc

        
    
# %%
# for i in range(2,len(riverQ)):
#     csoBool = distances['HubName'].eq(riverQ["beregningspunktlokalid"].iloc[i])
#     isThereCSO = csoBool.sum() == 1 # check if there is a CSO
#     if isThereCSO:
#         idxCSO = csoBool[csoBool == True].index
#         #print(distances.iloc[idxCSO])
#         CSO_flux=0 #create an empty value for the MP flux from the CSO
#         CSO_Qtot=0 #create an empty value for the total CSO flow added to the river
#         for j in range(1,len(idxCSO)): #loop over the connected CSOs
#             V_CSO= distances["Vandmaengd"][idxCSO[j]]*theta # calculate the volume of the single discharge event
#             Q_CSO=V_CSO/t_CSO # calculate the flow of the single discharge event
            
#             CSO_flux=CSO_flux+Q_CSO*CSO_conc
#             CSO_Qtot= CSO_Qtot+Q_CSO
#         riverQ['CSOflow'][i]= riverQ['CSOflow'][i-1]+CSO_Qtot
#         # calculate concentration for the node
#         riverC['concentration'][i]=(riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
#         riverQ['CSOflow'][i-1])+ CSO_flux)/ (riverQ['water flow (m^3/s)'][i]+
#         riverQ['CSOflow'][i])
#     else: # there are no CSO connected to the node
#         # calculate flow added to the river by the CSO (no addition)
#         riverQ['CSOflow'][i]= riverQ['CSOflow'][i-1]
#         # calculate concentration for the node
#         riverC['concentration'][i] =riverC['concentration'][i-1]*(riverQ['water flow (m^3/s)'][i-1]+
#         riverQ['CSOflow'][i-1])/ (riverQ['water flow (m^3/s)'][i]+
#         riverQ['CSOflow'][i])
   
# # EQS_exc= riverC["conc"]>EQS_exc
