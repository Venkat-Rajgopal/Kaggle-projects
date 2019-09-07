import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # data preparation

    # Customer id and Total charges is delt with separately.
    df.drop(['customerID'], axis = 1, inplace = True)
    df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')


    # df.dtypes reveals that all the fetures are set as object this needs to be corrected.
    # get all categorical fetures from the data and comvert them into binary
    # numerical cols are all the others. 

    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or c == 'SeniorCitizen']
    num_cols = [x for x in df.columns if x not in cat_cols]

    # categorize for uniques > 2 and others do a one hot encoding
    # make a categorial df for this purpose. This will be joined with numerical df
    df_cat = df[cat_cols]
    for col in cat_cols:
        if df_cat[col].nunique() == 2:
            df_cat[col], _ = pd.factorize(df_cat[col])
        else:
            df_cat = pd.get_dummies(df_cat, columns= [col])


    # Normalize the dataframe which contains numerical cols 
    df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')))

    # Join 
    df_final = pd.concat([df_std, df_cat], axis=1)
    print('Final processed data dimensions:', df_final.shape)

    return df_final