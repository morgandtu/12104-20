# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
riverC.reset_index()
EQS_exc= riverC.copy()

# %%
StringName=riverQ['beregningspunktlokalid'].str.split('_').str[-1]
StringName = StringName.astype(float)

# %%
type(StringName)

# %%
riverQ["distance"]= StringName - 3687

# %%
riverC['concentration'] = 0

# %%
# starting the simple advection-dilution model 
#riverC[0]= C0 #initial concentration
#CSO_conc= XX # assign the MP concentration in the CSO water

# %%
riverQ["beregningspunktlokalid"].

# %%
distances

# %%
distances["HubName"]

# %%
distances["HubName"].str.contains(riverQ["beregningspunktlokalid"].iloc[i]).sum()

# %%
for i in range(len(riverQ)):
    csoBool = distances["HubName"].str.contains(riverQ["beregningspunktlokalid"].iloc[i])
    isThereCSO = csoBool.sum() == 1 # check if there is a CSO
    if isThereCSO:
        idxCSO = csoBool[csoBool == True].index
        #print(distances.iloc[idxCSO])
        CSO_flux=0 #create an empty value for the MP flux from the CSO
        CSO_Qtot=0 #create an empty value for the total CSO flow added to the river
        for j in range(len(idxCSO)) #loop over the connected CSOs
            V_CSO= distances["Vandmaengd"][idxCSO[j]]*theta # calculate the volume of the single discharge event
            Q_CSO=V_CSO/t_CSO # calculate the flow of the single discharge event
            
            CSO_flux=CSO_flux+Q_CSO*CSO_conc
            CSO_Qtot= CSO_Qtot+Q_CSO
            riverQ['Qadded'][i]= riverQ['Qadded'][i-1]+CSO_Qtot
            # calculate concentration for the node
            riverC['conc'][i]=(riverC['conc'][i-1]*(riverQ['flow'][i-1]+
            riverQ['Qadded'][i-1])+ CSO_flux)/ (riverQ['flow'][i]+
            riverQ['Qadded'][i])
    else: # there are no CSO connected to the node
        # calculate flow added to the river by the CSO (no addition)
        riverQ['Qadded'][i]= riverQ['Qadded'][i-1]
        # calculate concentration for the node
        riverC['conc'][i] =riverC['conc'][i-1]*(riverQ['flow'][i-1]+
        riverQ['Qadded'][i-1])/ (riverQ['flow'][i]+
        riverQ['Qadded'][i])
   
EQS_exc= riverC["conc"]>EQS_exc



