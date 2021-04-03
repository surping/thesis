import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.pyplot import hist
from sklearn.neighbors import NearestNeighbors
import time

# load CSV
DapHosp_1 = pd.read_excel('/Users/user/Documents/thesis/DAP_cancer_events_hosp.xls', sheet_name='938 hospitals')
DapHosp_2 = pd.read_excel('/Users/user/Documents/thesis/DAP_cancer_events_hosp.xls', sheet_name='137 AMC_NCI')
# DapHosp = pd.read_csv('https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv')
DapHosp = pd.concat([DapHosp_1, DapHosp_2])
DapHosp.rename(columns={'Provider ID': 'Provider_ID'}, inplace=True)

# check for duplicates. No duplicates in the dataset
# no need to union both sheets. the second sheet is subset of first
DapHosp = DapHosp.drop_duplicates()

# the row with provider_id = 999999 is an aggregation. remove it
DapHosp = DapHosp[DapHosp.Provider_ID != 999999]

# removing negative values found in the dataset
# will not fill nan on purpose. they don seem to occur accidentally
# filled negative values with median. assigned dummy value 10mil in order to avoid fill nan.
DapHosp = DapHosp.fillna(1000000)
DapHosp[DapHosp.iloc[:, 4:38]  < 0] = np.nan
DapHosp = DapHosp.fillna(DapHosp.median())
DapHosp[DapHosp == 1000000] = np.nan
DapHosp = DapHosp.iloc[:, :] 

# pivot the dataset to work with z-score function
DapHospPvt = DapHosp.melt(id_vars=['Provider_ID', 'Hospital Name', 'City', 'State'], 
        var_name='Category', 
        value_name='Value')

# Delete completely rows with null values in column 'Percent of cancer patients dying in hospital (2003-07) Point estimate'
Providers_to_del = DapHospPvt[ (DapHospPvt['Category'] == 'Percent of cancer patients dying in hospital (2003-07)\nPoint estimate') & DapHospPvt['Value'].isna()]
DapHospPvt = pd.merge(DapHospPvt, Providers_to_del, on=['Provider_ID'], how="outer", indicator=True).query('_merge=="left_only"')
# rename columns
DapHospPvt = DapHospPvt.drop(['Hospital Name_y', '_merge', 'City_y', 'State_y', 'Category_y', 'Value_y'], axis = 1) 
DapHospPvt = DapHospPvt.rename({'Value_x': 'Value', 'Hospital Name_x': 'Hospital Name', 'City_x': 'City', 'State_x': 'State', 'Category_x': 'Category'}, axis=1)

# keep only point estimate rows ie: columns on original excel
searchfor = ['estimate', 'Number of deaths']
DapHospPvt = DapHospPvt[DapHospPvt['Category'].str.contains('|'.join(searchfor))]

# END of general data validation

# START statistical method for outliers for specific groups
start = time.time()

# remove nans temporarily to work with z-score function
DapHospPvt_notna = DapHospPvt[DapHospPvt['Value'].notna()]

# remove outliers with z-score < 4. Outlier found only on column 'Number of deaths among cancer patients assigned to hospital (2003-07)'
# z-score goes up to 7. I believe that no rows should be removed. But i removed z-score <4 for the thesis' needs
# will keep outliers and will do left exception join
DapHospPvt_notna = DapHospPvt_notna[DapHospPvt_notna.groupby(['Category']).Value.transform(lambda x : stats.zscore(x,ddof=1))>4]

# re-assemble datasets and exclude outliers
DapHospPvt_stats = pd.merge(DapHospPvt, DapHospPvt_notna, on=['Provider_ID', 'Hospital Name', 'City', 'State', 'Category'], how="outer", indicator=True).query('_merge=="left_only"')
# rename columns
DapHospPvt_stats = DapHospPvt_stats.drop(['_merge', 'Value_y'], axis = 1) 
DapHospPvt_stats = DapHospPvt_stats.rename({'Value_x': 'Value'}, axis=1)

end = time.time()

# plotting
# keep only deaths to plot
DapHospPvt_deaths = DapHospPvt_stats[DapHospPvt_stats['Category'].str.contains('Number of deaths', na=False)]
DapHospPvt_deaths = DapHospPvt_deaths.groupby(['State'])['Value'].sum().reset_index()

x = DapHospPvt_deaths.Value.to_numpy()
y = DapHospPvt_deaths.State.to_numpy()

def absolute_value(val):
    a  = np.round(val/100.*x.sum(), 0)
    return a

plt.pie(x, labels = y, autopct=absolute_value, shadow=True)
plt.title("Cancer deaths per state", bbox={'facecolor':'0.8', 'pad':5})

plt.show()
# END statistical method for outliers for specific groups

###################################################
###################################################

# START alorithmical method for outliers for specific groups
start_algo = time.time()

DapHospPvt_Algo = pd.DataFrame(DapHospPvt,columns=['State', 'Category', 'Value']) 

# change State and Category to custom category id. help for plot and algorithm
col1 = DapHospPvt_Algo["Category"].astype(str) 
DapHospPvt_Algo['Category_ID'] = (col1).astype(str).rank(method='dense', ascending=False).astype(int)

# keep relationship of category_id and indicator
DapHospPvt_desc = DapHospPvt_Algo[['State', 'Category', 'Value', 'Category_ID']]
DapHospPvt_Algo = DapHospPvt_desc[['Category_ID','Value']]

DapHospPvt_Algo = DapHospPvt_Algo[DapHospPvt_Algo['Value'].notna()]
DapHospPvt_Algo_grouped = DapHospPvt_Algo.groupby('Category_ID')

for group_name, df_group in DapHospPvt_Algo_grouped:

    for row_index, row in df_group.iterrows():

        X = df_group.values[:,1:] 
        # instantiate model
        nbrs = NearestNeighbors(n_neighbors = 3)
        # fit model
        nbrs.fit(X)
        # distances and indexes of k-neaighbors from model outputs
        distances, indexes = nbrs.kneighbors(X)
        # plot mean of k-distances of each observation
        plt.plot(distances.mean(axis =1))
        # visually determine cutoff values > 1.5 for case category 36
        outlier_index = np.where(distances.mean(axis = 1) > 0.25)
        # filter outlier values
        outlier_values = DapHospPvt_Algo.iloc[outlier_index]

DapHospPvt_desc = pd.merge(DapHospPvt_desc, outlier_values, on=['Category_ID','Value'], how="outer", indicator=True).query('_merge=="left_only"')
DapHospPvt_desc = DapHospPvt_desc.drop(['_merge'], axis = 1) 

end_algo = time.time()

del DapHospPvt_desc['Category_ID']

# plotting
# keep only deaths to plot
DapHospPvt_algo_deaths = DapHospPvt_desc[DapHospPvt_desc['Category'].str.contains('Number of deaths', na=False)]
DapHospPvt_algo_deaths = DapHospPvt_algo_deaths.groupby(['State'])['Value'].sum().reset_index()

x = DapHospPvt_deaths.Value.to_numpy()
y = DapHospPvt_deaths.State.to_numpy()

def absolute_value(val):
    a  = np.round(val/100.*x.sum(), 0)
    return a

plt.pie(x, labels = y, autopct=absolute_value, shadow=True)
plt.title("Cancer deaths per state", bbox={'facecolor':'0.8', 'pad':5})

plt.show()
# END alorithmical method for outliers for specific groups

print(DapHospPvt_deaths)
print(DapHospPvt_algo_deaths)
print(end - start)
print(end_algo - start_algo) 



