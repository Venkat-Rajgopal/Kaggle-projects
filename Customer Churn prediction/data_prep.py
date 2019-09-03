import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
style.use('fivethirtyeight')

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# data preparation

# df.dtypes reveals that all the fetures are set as object this needs to be corrected.
# get all categorical fetures from the data and comvert them into binary
# numerical cols are all the others. 

df.drop(['customerID'], axis = 1, inplace = True)
df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')

cat_cols = [c for c in df.columns if df[c].dtype == 'object' or c == 'SeniorCitizen']
num_cols = [x for x in df.columns if x not in cat_cols]

# categorize for uniques > 2 and others do a one hot encoding
# make a categorial df for this purpose
df_cat = df[cat_cols]
for col in cat_cols:
    if df_cat[col].nunique() == 2:
        df_cat[col], _ = pd.factorize(df_cat[col])
    else:
        df_cat = pd.get_dummies(df_cat, columns= [col])


# Normalize the dataframe which contains numerical cols 
df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')))

df_final = pd.concat([df_std, df_cat], axis=1)
print('Final processed data dimensions:', df_final.shape)

