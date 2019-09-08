import os
from os import chdir, getcwd
import warnings

wd=getcwd()
chdir(wd)

import numpy as np 
from data_prep import prepare_data
from cm_plot import plot_cm

# Modelling imports from Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.svm import SVC

# Visualization
import matplotlib.style as style
import matplotlib.pyplot as plt
style.use('seaborn')


df_final = prepare_data()

# Train Test Split and set targets
train, test = train_test_split(df_final, test_size = .2, random_state = 10)

feats = [c for c in train.columns if c not in ['Churn']]
target = ['Churn']

train_x = train[feats]
train_y = np.ravel(train[target])
test_x = test[feats]
test_y = np.ravel(test[target])


clf = SVC()
param_grid = rbf_grid =  {'C': np.logspace(-4, 1, 10, base=2), 'gamma': np.logspace(-6, 2, 10, base=2)}
metrics = ['roc_auc', 'accuracy']

gs = GridSearchCV(clf, param_grid = param_grid, cv = 5, scoring = metrics ,verbose=1, refit = 'roc_auc')
gs.fit(train_x, train_y)

[(m, gs.cv_results_['mean_test_{}'.format(m)][gs.best_index_]) for m in metrics]

preds = gs.predict(test_x)
probs = gs.predict_proba(test_x)

print ("Accuracy : ", accuracy_score(test_y, preds))
print("Classification report : \n", classification_report(test_y, preds))

# confusion matrix
cm = confusion_matrix(test_y, preds)

# roc_auc_score
model_roc_auc = roc_auc_score(test_y, preds) 
print('ROC_AUC score: ' ,model_roc_auc)
fpr,tpr,thresholds = roc_curve(test_y, probs[:,1])