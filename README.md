# **Credit Risk : Data Analysis in Python and Visualization in Tableau**
Author: Akshay B

## 1. Project Objective
The aim of this project is to mainly to predict whether a loan applicant will default or not before the loan is processed, this project was done basically to see if the
can be replicated in payment and ecommerce industries so that the merchant can be classified as fraud or not before he is onboarded. 

## 2. Project Description
The data used for the purpose of this project was taken from Kaggle. The dataset has 32581 rows and 12 columns. The data had its limitation of having blank rows. 
The columns were skewed, and the dataset also had the problem that is commonly seen across classification type datasets, the target variable was imbalance. So initially 
logistic regression model will be used with the imbalanced data for prediction and later the dataset will be balanced out and then will be fed to the logistic regression model

## 3. Cleaning the data

Exploring the data 
```
loan_risk.shape
loan_risk.dtypes
loan_risk.describe()
```
Rearranging the columns
```
loan_risk.columns
loan_risk = loan_risk.reindex(columns=['person_age', 'person_income', 'person_home_ownership',
           'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
           'loan_int_rate','loan_percent_income',
           'cb_person_default_on_file', 'cb_person_cred_hist_length','loan_status'])
```
Dealing with outliers

`Outliers were removed using the IOR (Inter Quartile range) method`

Dealing with missing values
```
loan_risk["person_emp_length"].fillna(loan_risk["person_emp_length"].mean(),inplace=True)
loan_risk["loan_int_rate"].fillna(loan_risk["loan_int_rate"].mean(),inplace=True)
```

Transforming categorical columns to numeric
```
# we shall use factorize funtion for conversion, loan_risk.select_dtypes(include=object).columns will give you all the object type columns
for i in loan_risk.select_dtypes(include=object).columns:
  loan_risk[i]=pd.factorize(loan_risk[i])[0]
```
## 4. Plotting graphs

Distribution of object type variables
```
fig,axes=plt.subplots(3,3,figsize=(18,10))
fig.suptitle("Distributions")
         
sns.histplot(ax=axes[0,0],data=loan_risk["person_age"],kde=True);
sns.boxplot(ax=axes[0,1],data=loan_risk["person_income"]);
sns.histplot(ax=axes[0,2],data=loan_risk["person_emp_length"],kde=True);
sns.violinplot(ax=axes[1,0],data=loan_risk["loan_amnt"]);
sns.boxplot(ax=axes[1,1],data=loan_risk["loan_int_rate"]);
sns.violinplot(ax=axes[1,2],data=loan_risk["loan_percent_income"]);
sns.histplot(ax=axes[2,1],data=loan_risk["cb_person_cred_hist_length"],kde=True);
# I have dropped all the object dtypes and target variable
```
![distribution](https://user-images.githubusercontent.com/86428423/173227682-9f542df2-4b19-46e1-aa4a-2ea2481df468.png)

Plotting the target variable to see the imbalance in the dataset
```
#loan_risk["loan_status"].value_counts(1) --> will give the percent of 0s and 1s in the target column
#I have used semicolon at the end to remove the text that appeears before the plot
plt.pie(loan_risk["loan_status"].value_counts(1),labels=["0s","1s"],autopct='%.2f');
```
![imbalance](https://user-images.githubusercontent.com/86428423/173227807-3f70cd70-2339-4121-8a80-d75c1b3377a0.png)

## 5. Designing the model

I have used Logistic regression model

Splitting and normalizing the dataset
```
x=loan_risk.drop(["loan_status"],axis=1)
y=loan_risk["loan_status"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=25) #using random sate to get the same values every time query is run and train:test ratio = .75:.25
xtrainnorm=pd.DataFrame(preprocessing.normalize(xtrain,axis=0),columns=xtrain.columns)
xtestnorm=pd.DataFrame(preprocessing.normalize(xtest,axis=0),columns=xtest.columns)
```
Loading imbalanced data into the model
```
#Logistic Regression model and predicting ytest values
m=LogisticRegression(random_state=1)
m.fit(xtrainnorm,ytrain)
ypred=m.predict(xtestnorm)
```
`even if the accuracy of the model is 78% our model is predicting that all the output as 0 and it fails to predict any 1's. This is due to a case that we had seen earlier that our target variable is imbalanced.`

Ttying SMOTE oversampling and NearMiss undersampling for tackling imbalance in the target variable
```
sm=SMOTE(random_state=2)
xover,yover=sm.fit_resample(xtrainnorm,ytrain)
```
`Accuracy 69%`

![smote](https://user-images.githubusercontent.com/86428423/173228161-1e7d33e3-4e97-4dbc-95c3-975f45a431db.png)

```
nm=NearMiss()
xunder,yunder=nm.fit_resample(xtrainnorm,ytrain)
```
`Accuracy 69%`

