# Read dataframes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def get_data_split():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # drop id column
    train = train.drop(['id'], axis=1)
    test = test.drop(['id'], axis=1)

    # reform targets as one hot encoded vectors. 
    #y = train['target']
    #y = y.values.reshape(-1,1)
    #cat = OneHotEncoder()
    #y = cat.fit_transform(y).toarray()

    y = pd.Categorical(train['target']).codes

    # drop target vector 
    train = train.drop(['target'], axis=1)

    # Split training samples for train and val. 
    x_train, x_val, y_train, y_val = train_test_split(train, y, test_size = 0.2)
    print('Train size:', x_train.shape)
    print('Val size:', x_val.shape)
    print('Train labels:', y_train.shape)
    print('Val labels:', y_val.shape)

    return x_train, x_val, y_train, y_val
