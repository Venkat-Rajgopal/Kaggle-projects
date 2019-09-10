import os
from os import chdir, getcwd
import warnings

wd=getcwd()
chdir(wd)

import numpy as np 
import pandas as pd
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

            #print(self.fixed_params)
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
                fpr_,tpr_, _ = roc_curve(self.test.Churn, prob)

                roc_auc.append(res)
                fpr.append(fpr_)
                tpr.append(tpr_)
            else:
                # All other classifiers
                acc.append((m, eval("{}_score".format(m))(self.test.Churn, preds)))

        
        self.cm = confusion_matrix(self.test.Churn, preds)

        return self.cm, fpr, tpr, roc_auc

        #return preds, roc_auc, fpr, tpr, cm , acc


    # Plot a confusion matrix for the model
    def plot_results(self):
        out_path = os.path.abspath('plots')
        fig = plt.figure(figsize=(5, 5)) 
        plot_cm(self.cm, classes=np.unique(self.train.Churn), mtd = self.estimator.__name__)
        fig.savefig(os.path.join(out_path, self.estimator.__name__ + '_cm' + '.png'), bbox_inches='tight', dpi=100)
        plt.show()



df_final = prepare_data()
lr_grid = {'C':  np.logspace(-4, 4, 100, base=10) }
rbf_grid =  {'C': np.logspace(-4, 1, 10, base=2), 'gamma': np.logspace(-6, 2, 10, base=2)}
metrics = ['roc_auc', 'accuracy']


def get_results(data):

    # Fit training model
    model = Train(classifier = LogisticRegression, data = data, test_percent = 0.2, metrics = metrics)

    # Cross validate and get opt parameteres
    model.cross_val(fit_metric = 'roc_auc', n_val = 3, grid_params = lr_grid)

    # Execute and evaluate on train and test
    model.model_train()
    model.plot_results()

#get_results(data = df_final)

def svm(data):

    # Fit training model
    print('Fitting model')
    model = Train(classifier = SVC, data = data, test_percent = 0.2, metrics = metrics, fixed_params= {'kernel': 'rbf', 'probability': True})

    # Cross validate and get opt parameteres
    print('Getting params')
    model.cross_val(fit_metric = 'roc_auc', n_val = 3, grid_params = rbf_grid)

    # Execute and evaluate on train and test
    print('Training and evaluation')
    return model.model_train()

#preds, roc_auc, fpr, tpr, cm, acc  = svm(data = df_final)



# Initiate classifiers
classifiers = [LogisticRegression, SVC]
# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

grids = {'lr_grid': lr_grid, 'rbf_grid': rbf_grid}

# Train all models and record results 
for clf in classifiers:
    print('Computing ', clf.__name__, '\n')

    if clf == LogisticRegression:
        use_grid = grids['lr_grid']
        fixed_params = {}
    if clf == SVC:
        use_grid = grids['rbf_grid']
        fixed_params = {'kernel': 'rbf', 'probability': True}

    model = Train(classifier = clf, data = df_final, test_percent = 0.2, metrics = metrics, fixed_params = fixed_params)

    model.cross_val(fit_metric = 'roc_auc', n_val = 3, grid_params = use_grid)
    cm, fpr, tpr, roc_auc = model.model_train()
    model.plot_results()
    
    # append results to the table
    result_table = result_table.append({'classifiers':clf.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':roc_auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

# Plot combined ROC 
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
fig.savefig('plots/multimodel_roc_curve.png')