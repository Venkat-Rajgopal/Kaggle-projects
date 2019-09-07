import os
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import numpy as np 
import pandas as pd 

# Modelling imports from Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from cm_plot import plot_cm

from data_prep import prepare_data
# Visualization
import matplotlib.style as style
import matplotlib.pyplot as plt
style.use('seaborn')

# ------------------------------------------------------------------------
df_final = prepare_data()

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
clf = LogisticRegression(solver = 'liblinear')
param_grid = {'C':  np.logspace(-4, 4, 100, base=10) }
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

# ------------------------------------------------------------------------
# Plot  confusion matrix and roc curve
out_path = os.path.abspath('plots')


fig = plt.figure(figsize=(10, 5)) 
plt.subplot(1,2,1)
plot_cm(cm, classes=np.unique(df.Churn), mtd = 'Logistic')
plt.subplot(1,2,2)
#plt.plot(fpr, tpr, linestyle = '-', color = "royalblue", linewidth = 2)
plt.plot(fpr, tpr, color='royalblue', label='{} {}'.format('Logistic_regression AUC:',np.round(model_roc_auc,3)))
plt.plot([0, 1], [0, 1], linestyle='--', color='darkorange')
plt.legend(loc="lower right")

fig.savefig(os.path.join(out_path, 'log_reg_cm_roc.png'), bbox_inches='tight', dpi=100)

plt.show()



