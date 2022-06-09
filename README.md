# **Credit Risk : Data Analysis in Python and Visualization in Tableau**
Author: Akshay B

## Project Objective
The aim of this project is to mainly to predict whether a loan applicant will default or not before the loan is processed, this project was done basically to see if the
can be replicated in payment and ecommerce industries so that the merchant can be classified as fraud or not before he is onboarded. 

## Project Description
The data used for the purpose of this project was taken from Kaggle. The dataset has 32581 rows and 12 columns. The data had its limitation of having blank rows. 
The columns were skewed, and the dataset also had the problem that is commonly seen across classification type datasets, the target variable was imbalance. So initially 
logistic regression model will be used with the imbalanced data for prediction and later the dataset will be balanced out and then will be fed to the logistic regression model

## Process

Exploring the data 
* loan_risk.shape
* loan_risk.dtypes
* loan_risk.describe()

Rearranging the columns
* loan_risk.columns
* loan_risk = loan_risk.reindex(columns=['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate','loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length','loan_status'])


