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
# ------------------------------------------------------------------------
df_final = prepare_data()

class Train():
    def __init__(self, classifier, data, test_percent, metrics, fixed_params = {}, random_seed = 50):
        
        # --- set parameters ----
        self.estimator = classifier
        self.seed = random_seed
        self.metrics = metrics

        # save hyperparameters here
        self.hyperparametets = {}
        self.fixed_params = fixed_params
        self.fixed_params['random_state'] = random_seed

        # solver update for sklearn logistic regression classifier
        if classifier == LogisticRegression:
            self.fixed_params['solver'] = 'lbfgs'

        # --- Data partition ----
        self.train, self.test = train_test_split(data, test_size = test_percent, random_state = self.seed)
        # set features and targets 
        self.feats = [f for f in self.train.columns if f not in ['Churn']]


    # Train with Cross validation and get the best parameters. 
    def cross_val(self, fit_metric, n_val, grid_params):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            gs = GridSearchCV(self.estimator(**self.fixed_params), param_grid = grid_params, cv = n_val, scoring = self.metrics, refit = fit_metric)
            gs.fit(self.train[self.feats], self.train.Churn)
            self.hyperparameters = gs.best_params_

            #result = [(m, gs.cv_results_['mean_test_{}'.format(m)][gs.best_index_]) for m in self.metrics] 

            return print('Hyperparamater selected: ', self.hyperparameters)

    # Train based on the best hyperparameter
    def model_train(self):

        # call the fixed and hyperparameters here and fit into the classifier
        params = {**self.hyperparameters, **self.fixed_params}
        clf = self.estimator(**params)
        clf.fit(self.train[self.feats], self.train.Churn)

        # get predictions and probabilities of the positive class
        preds = clf.predict(self.test[self.feats])
        prob = clf.predict_proba(self.test[self.feats])[:,1]

        # Get results based on the evaluation metric
        roc_auc = list()
        fpr = list()
        tpr = list()
        acc = list()

        for m in self.metrics:
            if m == 'roc_auc':
                res = roc_auc_score(self.test.Churn, prob)
                fpr_,tpr_,thresholds_ = roc_curve(self.test.Churn, prob)

                roc_auc.append(res)
                fpr.append(fpr_)
                tpr.append(tpr_)
            else:
                # All other classifiers
                acc.append((m, eval("{}_score".format(m))(self.test.Churn, preds)))

        
        cm = confusion_matrix(self.test.Churn, preds)

        return preds, roc_auc, fpr, tpr, cm , acc



lr_grid = {'C':  np.logspace(-4, 4, 100, base=10) }
rbf_grid =  {'C': np.logspace(-4, 1, 10, base=2), 'gamma': np.logspace(-6, 2, 10, base=2)}
metrics = ['roc_auc', 'accuracy']


def get_results(data):

    # Fit training model
    model = Train(classifier = LogisticRegression, data = data, test_percent = 0.2, metrics = metrics)

    # Cross validate and get opt parameteres
    model.cross_val(fit_metric = 'roc_auc', n_val = 3, grid_params = lr_grid)

    # Execute and evaluate on train and test
    return model.model_train()


preds, roc_auc, fpr, tpr, cm, acc  = get_results(data = df_final)

def svm(data):

    # Fit training model
    print('Fitting model')
    model = Train(classifier = SVC, data = data, test_percent = 0.2, metrics = metrics, fixed_params= {'kernel': 'rbf', 'probability': True})

    # Cross validate and get opt parameteres
    print('Getting params')
    model.cross_val(fit_metric = 'roc_auc', n_val = 4, grid_params = rbf_grid)

    # Execute and evaluate on train and test
    print('Training and evaluation')
    return model.model_train()

#preds, roc_auc, fpr, tpr, cm, acc  = svm(data = df_final)

out_path = os.path.abspath('plots')
fig = plt.figure(figsize=(10, 5)) 
plt.subplot(1,2,1)
plot_cm(cm, classes=np.unique(df_final.Churn), mtd = 'Log reg')
plt.subplot(1,2,2)
plt.plot(fpr[0], tpr[0], color='royalblue', label='{} {}'.format('LR AUC:',np.round(roc_auc,3)))
plt.plot([0, 1], [0, 1], linestyle='--', color='darkorange')
plt.legend(loc="lower right")
fig.savefig(os.path.join(out_path, 'LRhypparm_cm_roc.png'), bbox_inches='tight', dpi=100)
plt.show()