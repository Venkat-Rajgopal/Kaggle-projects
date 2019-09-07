import os
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import numpy as np 
from data_prep import prepare_data

# Modelling imports from Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve


# ------------------------------------------------------------------------
df_final = prepare_data()

class Train():
    def __init__(self, classifier, data, test_percent, metrics, random_seed = 50):
        
        # --- set parameters ----
        self.estimator = classifier
        self.seed = random_seed
        self.metrics = metrics
        # save hyperparameters here
        self.hyperparametets = {}

        # --- Data partition ----
        self.train, self.test = train_test_split(data, test_size = test_percent, random_state = self.seed)
        # set features and targets 
        self.feats = [f for f in self.train.columns if f not in ['Churn']]


    # Train with Cross validation and get the best parameters. 
    def cross_val(self, fit_metric, n_val, grid_params):
        gs = GridSearchCV(self.estimator, param_grid = grid_params, cv = n_val, scoring = self.metrics , verbose=1, refit = fit_metric)
        gs.fit(self.train[self.feats], self.train.Churn)
        self.hyperparameters = gs.best_params_

        #result = [(m, gs.cv_results_['mean_test_{}'.format(m)][gs.best_index_]) for m in self.metrics] 

        return print('Hyperparamater selected: ', self.hyperparameters)

    # Train based on the best hyperparameter
    def model_train(self):
        params = {**self.hyperparameters}
        #clf = self.estimator(**params).fit(self.train[self.feats], self.train.Churn)
        clf = self.estimator(params)
        clf.fit(self.train[self.feats], self.train.Churn)

        # get predictions and probabilities of the positive class
        preds = clf.predict(self.test[self.feats])
        prob = clf.predict_proba(self.test.Churn)[:, 1]

        # Get results based on the evaluation metric
        roc_auc = list()
        fpr_tpr = list()

        for m in self.metrics:
            if m == 'roc_auc':
                res = roc_auc_score(self.test.Churn, prob)
                fpr,tpr,thresholds = roc_curve(self.test.Churn, prob)

                roc_auc.append(res)
                fpr_tpr.append(fpr,tpr,thresholds)

        

        return preds, roc_auc, fpr_tpr



param_grid = {'C':  np.logspace(-4, 4, 100, base=10) }
metrics = ['roc_auc', 'accuracy']



def get_results(data):
    #model = Train(classifier = LogisticRegression(solver = 'liblinear'), data = df_final, test_percent = 0.2, metrics = metrics)
    model = Train(classifier = LogisticRegression, data = df_final, test_percent = 0.2, metrics = metrics)

    result = model.cross_val(fit_metric = 'roc_auc', n_val = 5, grid_params = param_grid)

    model.model_train()


get_results(data = df_final)