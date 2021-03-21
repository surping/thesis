import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import time

import re
import nltk
from string import punctuation
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
# nltk.download()

# START function and initializations for NLP preprocess 

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

# END function and initializations for NLP preprocess 

#load CSV
# bigCts = pd.read_csv('/Users/user/Documents/thesis/Big_Cities_Health_Data_Inventory.csv')
bigCts = pd.read_csv("https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv")
# bigCts_Full = pd.read_csv("https://raw.githubusercontent.com/surping/thesis/master/Big_Cities_Health_Data_Inventory.csv")
bigCts = pd.DataFrame(bigCts,columns=['Indicator Category', 'Indicator', 'Year', 'Gender', 'Race/ Ethnicity', 'Place', 'Value', 'Source', 'Methods'])

# START NLP preprocess on specific columns

bigCts['Source'] = bigCts['Source'].map(lambda s:preprocess(s)) 
bigCts['Methods'] = bigCts['Methods'].map(lambda s:preprocess(s)) 

# END NLP preprocess on specific columns

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
bigCts = bigCts.drop(['Value_y', '_merge', 'Source_y', 'Methods_y'], axis = 1) 
bigCts = bigCts.rename({'Value_x': 'Value', 'Source_x': 'Source', 'Methods_x': 'Methods'}, axis=1)
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

bigCts2012.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)
bigCts2013.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)
bigCts2014.boxplot(column='Value', by=['Indicator', 'Gender', 'Race/ Ethnicity'], grid=False, rot = 30, fontsize=6)

plt.show()
#END statistical method for outliers for specific groups

###################################################
###################################################

#START alorithmical method for outliers for specific groups
start_algo = time.time()

bigCts_ctg = bigCts[['Indicator', 'Year', 'Gender', 'Race/ Ethnicity', 'Value']]

col1 = bigCts_ctg["Indicator"].astype(str) 
col2 = bigCts_ctg["Gender"].astype(str) 
col3 = bigCts_ctg["Race/ Ethnicity"].astype(str)
col4 = bigCts_ctg["Year"].astype(str)
# change Indicator to custom category id. help for plot and algorithm
bigCts_ctg['Category_ID'] = (col1+col2+col3+col4).astype(str).rank(method='dense', ascending=False).astype(int)

# keep relationship of category_id and indicator
bigCts_ctg_desc = bigCts_ctg[['Category_ID', 'Indicator', 'Year', 'Gender', 'Race/ Ethnicity', 'Value']]
bigCts_ctg = bigCts_ctg_desc[['Category_ID','Value']]

# eliminate groups with less than 4 records, otherwise neighbors not working
bigCts_ctg = bigCts_ctg.groupby('Category_ID').filter(lambda group: len(group) > 3).sort_index( ascending=False)

# below i choose one category for test to apply alogrithmical outlier elimination
# bigCts_ctg = bigCts_ctg[(bigCts_ctg['Category_ID'] == 36) | (bigCts_ctg['Category_ID'] == 37)]

bigCts_ctg_grouped = bigCts_ctg.groupby('Category_ID')

for group_name, df_group in bigCts_ctg_grouped:

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
        outlier_index = np.where(distances.mean(axis = 1) > 2)
        # filter outlier values
        outlier_values = bigCts_ctg.iloc[outlier_index]

bigCts_ctg_desc = pd.merge(bigCts_ctg_desc, outlier_values, on=['Category_ID','Value'], how="outer", indicator=True).query('_merge=="left_only"')
bigCts_ctg_desc = bigCts_ctg_desc.drop(['_merge'], axis = 1) 

end_algo = time.time()
#END alorithmical method for outliers for specific groups


# pd.set_option('display.max_rows', None)
# print(bigCts)
# print(bigCts_ctg)
print(bigCts2012)
print(bigCts2013)
print(bigCts2014)
print(bigCts_ctg_desc)
print(end - start)
print(end_algo - start_algo)
plt.show()


