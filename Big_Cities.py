import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#load CSV
# bigCts = pd.read_csv('/Users/user/Documents/thesis/Big_Cities_Health_Data_Inventory.csv')
bigCts = pd.read_csv("https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv")
bigCts = pd.DataFrame(bigCts,columns=['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place','Value'])
#noticed by results: two rows had similar description. Fix below
bigCts['Indicator'] = bigCts['Indicator'].replace(['Opioid-Related Mortality Rate (Age-Adjusted; Per 100,000 people) *These data should not be compared across cities as they have different definitions.'],'Opioid-Related Mortality Rate (Age-Adjusted; Per 100,000 people) *These data should not be compared across cities as they have different definitions')

#find if there are null in columns that contain data we care about
bigCts_null_cols = pd.DataFrame(bigCts,columns=['Indicator Category','Indicator','Gender','Place','Value'])

#return the rows with the null columns
bigCts_null_cols = bigCts_null_cols[pd.isna(bigCts_null_cols).any(axis=1)]
bigCts_na_dist = bigCts_null_cols.drop_duplicates()

#fill nan Values with median
bigCts_toFill = pd.merge(bigCts,bigCts_na_dist,on=['Indicator Category','Indicator','Gender','Place'])
bigCts_toFill.Value_x = bigCts_toFill.groupby(['Indicator Category','Indicator','Gender','Place'])['Value_x'].apply(lambda x: x.fillna(x.median()))
bigCts_toFill.Value_x = bigCts_toFill.Value_x.fillna(bigCts_toFill.Value_x.median())
bigCts = bigCts.merge(bigCts_toFill, on=['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place'], how='left')
bigCts.Value = bigCts.Value.fillna(bigCts.Value_x)
bigCts = bigCts.drop(['Value_x','Value_y'], axis = 1) 

#drop duplicates with same value
bigCts = bigCts.drop_duplicates(subset=['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place','Value'], keep=False)
#remaining duplicates
bigCts_dups = bigCts[bigCts.duplicated(['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place'],keep=False)]
#remove duplicates / z-score method because some values are outliers
bigCts_dups = bigCts_dups[np.abs(stats.zscore(bigCts_dups['Value']))<1]
#remove duplicates / mean method
bigCts_dups = bigCts_dups.groupby(['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place']).mean().reset_index()
#remove from main dataframe the duplicates dataframe (left exception join)
bigCts = pd.merge(bigCts, bigCts_dups, on=['Indicator Category','Indicator','Year','Gender','Race/ Ethnicity','Place'], how="outer", indicator=True).query('_merge=="left_only"')
#rename columns
bigCts = bigCts.drop(['Value_y', '_merge'], axis = 1) 
bigCts = bigCts.rename({'Value_x': 'Value'}, axis=1)
#append duplicates subset to main dataframe
bigCts = pd.concat([bigCts, bigCts_dups]) 

print(bigCts)


