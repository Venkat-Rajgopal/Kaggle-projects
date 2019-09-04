# Predicting Churns for Telcom customers

## Customer Churns

Customer churn is the percentage of customers that stopped using a company's service during a certain time frame. In principle, these are 'lost customers' during a certain period. 

Losing customers is never a good news and this is more of a *hard truth* for the company. Also, not to mention that Customer churns is thus perhaps the most important metric for a growing company. 

Prediction customer churns thus is a critical step in where Data Science algorithms can significantly help retaining customers. 

Data is taken from the [Kaggle Competition](https://www.kaggle.com/blastchar/telco-customer-churn). 


## Methods and Results
The data consists of pre-labelled customers as 'Churn and Non-Churns'. 

A Logistic Regression binary classification is performed, resulting in `roc_auc` score of `0.71` shown in the below plot. 

![](/plots/log_reg-result.png)
