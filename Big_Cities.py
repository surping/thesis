import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import time
import re


#load CSV
# bigCts = pd.read_csv('/Users/user/Documents/thesis/Big_Cities_Health_Data_Inventory.csv')
bigCts = pd.read_csv("https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv")
bigCts_Full = pd.read_csv("https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv")
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
bigCts = pd.concat([bigCts, bigCts_dups]).reset_index(drop=True)

bigCts = bigCts[bigCts['Year'].isin(['2012','2013','2014']) & (bigCts['Gender'] != 'Both')  & (bigCts['Race/ Ethnicity'] != 'All')]

#START statistical method for outliers for specific groups
start = time.time()

bigCts2012 = bigCts[bigCts['Year'].isin(['2012'])]
bigCts2012 = bigCts2012.drop(['Indicator Category', 'Place', 'Year'], axis=1)
bigCts2012 = bigCts2012[bigCts2012.groupby(['Indicator', 'Gender', 'Race/ Ethnicity']).Value.transform(lambda x : stats.zscore(x,ddof=1))<3]

bigCts2013 = bigCts[bigCts['Year'].isin(['2013'])]
bigCts2013 = bigCts2013.drop(['Indicator Category', 'Place', 'Year'], axis=1)
bigCts2013 = bigCts2013[bigCts2013.groupby(['Indicator', 'Gender', 'Race/ Ethnicity']).Value.transform(lambda x : stats.zscore(x,ddof=1))<3]

bigCts2014 = bigCts[bigCts['Year'].isin(['2014'])]
bigCts2014 = bigCts2014.drop(['Indicator Category', 'Place', 'Year'], axis=1)
bigCts2014 = bigCts2014[bigCts2014.groupby(['Indicator', 'Gender', 'Race/ Ethnicity']).Value.transform(lambda x : stats.zscore(x,ddof=1))<3]

end = time.time()

# bigCts2012.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)
# bigCts2013.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)
# bigCts2014.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)

#END statistical method for outliers for specific groups

###################################################
###################################################

#START alorithmical method for outliers for specific groups


# bigCts_ctg = bigCts[['Indicator', 'Year', 'Gender', 'Race/ Ethnicity', 'Value']]

# col1 = bigCts_ctg["Indicator"].astype(str) 
# col2 = bigCts_ctg["Gender"].astype(str) 
# col3 = bigCts_ctg["Race/ Ethnicity"].astype(str)
# col4 = bigCts_ctg["Year"].astype(str)
# # change Indicator to custom category id. help for plot and algorithm
# bigCts_ctg['Category_ID'] = (col1+col2+col3+col4).astype(str).rank(method='dense', ascending=False).astype(int)

# # keep relationship of category_id and indicator
# bigCts_ctg_desc = bigCts_ctg[['Category_ID', 'Indicator', 'Year', 'Gender', 'Race/ Ethnicity', 'Value']]

# # keep only Category_ID and value to use in algorithm
# bigCts_ctg = bigCts_ctg_desc[['Category_ID','Value']]
# bigCts_ctg_list  = bigCts_ctg[['Category_ID']].drop_duplicates().reset_index(drop=True)

# # below i choose one category for test to apply alogrithmical outlier elimination
# bigCts_ctg = bigCts_ctg[bigCts_ctg['Category_ID'] == 36]

# X = bigCts_ctg.values
# # instantiate model
# nbrs = NearestNeighbors(n_neighbors = 2)
# # fit model
# nbrs.fit(X)
# # distances and indexes of k-neaighbors from model outputs
# distances, indexes = nbrs.kneighbors(X)
# # plot mean of k-distances of each observation
# plt.plot(distances.mean(axis =1))
# # visually determine cutoff values > 1.5 for case category 36
# outlier_index = np.where(distances.mean(axis = 1) > 1.8)
# # filter outlier values
# outlier_values = bigCts_ctg.iloc[outlier_index]

#END alorithmical method for outliers for specific groups

# START prepare Methods and Source for NLP


bigCts_Full = pd.DataFrame(bigCts_Full,columns=['Indicator', 'Source', 'Methods'])
bigCts_Full = bigCts_Full.iloc[1110:1125, :]
bigCts_Full['Methods'] = bigCts_Full['Methods'].apply(str)
bigCts_Full['Source'] = bigCts_Full['Source'].apply(str)
bigCts_Full['Methods'] = bigCts_Full['Methods'].map(lambda x: re.sub(r'[^a-zA-Z0-9. ]',r'',x))
bigCts_Full['Source'] = bigCts_Full['Source'].map(lambda x: re.sub(r'[^a-zA-Z0-9. ]',r'',x))

# END prepare Methods and Source for NLP

plt.show()
# pd.set_option('display.max_rows', None)
print(bigCts_Full)
# print(outlier_values)
# print(bigCts_ctg)
# print(bigCts_ctg_list)
# print(end - start)


