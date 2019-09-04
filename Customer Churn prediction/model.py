import os
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import numpy as np 
import pandas as pd 
import matplotlib.style as style
import seaborn as sns


# import gc
# import warnings

# Modelling imports from Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from cm_plot import plot_confusion_matrix


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
style.use('fivethirtyeight')

# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# Train Test Split and set targets
train, test = train_test_split(df_final, test_size = .2, random_state = 10)

feats = [c for c in train.columns if c not in ['Churn']]
target = ['Churn']

train_x = train[feats]
train_y = np.ravel(train[target])
test_x = test[feats]
test_y = np.ravel(test[target])
# ------------------------------------------------------------------------
# Train model and evaluate
clf = LogisticRegression().fit(train_x, train_y)
preds = clf.predict(test_x)
probs = clf.predict_proba(test_x)

print ("Accuracy : ", accuracy_score(test_y, preds))
print("Classification report : \n", classification_report(test_y, preds))

# confusion matrix
cm = confusion_matrix(test_y, preds)

# roc_auc_score
model_roc_auc = roc_auc_score(test_y, preds) 
print('ROC_AUC score: ' ,model_roc_auc)
fpr,tpr,thresholds = roc_curve(test_y, probs[:,1])


# Plot  confusion matrix and roc curve
out_path = os.path.abspath('plots')


fig = plt.figure(figsize=(8, 5)) 
plt.subplot(2,2,1);
plot_confusion_matrix(test_y, preds, classes=np.unique(df.Churn), title='Confusion matrix')

plt.subplot(2,2,2);
plt.plot(fpr, tpr, linestyle = '-', color = "royalblue", linewidth = 2)

fig.savefig(os.path.join(out_path, 'cm_roc.png'), bbox_inches='tight', dpi=100)


plt.show()