# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# pandas to work with data frames
import pandas as pd

# numpy import to perform numerical operations
import numpy as np

# to visualize data

import seaborn as sns

# partition of data
from sklearn.model_selection import train_test_split

# library for logistic regression
from sklearn.linear_model import LogisticRegression

#importing performance matrix - accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

#=========================
# @@@@ Importing data
#=========================

data_income = pd.read_csv('income.csv')

# copying data
data1 = data_income.copy()

#======================

# @@@ Exploratory data analysis

#================================

# 1 getting to know about data

#-----------------------------

# variable data types
data1.info()

# missing values
data1.isnull()

#total missing values
data1.isnull().sum()
 
# No missing values  we can see

# summary of the numerical values
num_summary = data1.describe()
print(num_summary)

# summary of the categorial values
cat_summary = data1.describe(include="O")
print(cat_summary)


#frequency of each categories
data1['JobType'].value_counts()
data1['occupation'].value_counts()

# unique values of each categorical variable
np.unique(data1['JobType'])
np.unique(data1['occupation'])


# Import Data again with ' ?' 

data1 = pd.read_csv('income.csv', na_values=[" ?"])

# 2. Data Preprocessing

#--------------------------------
# missing values
data1.isnull().sum()

# now we have 2 columns with missing values

# missing data frame with atleast one column value is missing

missing = data1[data1.isnull().any(axis=1)]

# There's no relation between the categorial variables, we delete all rows with atleast one NA value

data2 = data1.dropna(axis=0)

# correlation relation bw variables

correlation = data2.corr()
print(correlation)

#=====================

# Cross tables and data visualixation

#=========================

# Columns names

data2.columns

#gender proportion

gender_Prop = print(pd.crosstab(index=data2["gender"], columns='count', normalize=True))
pd.crosstab(index=data2["gender"], columns='count')
# gender vs salstat
data2.info()
gender_salstat = print(pd.crosstab(index=data2["gender"],columns=data2["SalStat"],margins=True,normalize='index'))


# Frequency distribution of salstat
sns.countplot(data2["SalStat"])

#Histograma of age
sns.distplot(data2["age"], bins=10, kde=False)


sns.boxplot("SalStat", "age", data=data2)


#bar plot Jobtype vs salstat

sns.countplot(y="JobType", hue="SalStat", data=data2)

print(pd.crosstab(index=data2["JobType"],columns=data2["SalStat"],margins=True,normalize='index'))


sns.countplot(y="EdType", hue="SalStat", data=data2)






# ======================================

# Logistic Regression

# =========================================

# reindexing the salstat variable to 0 and 1
np.unique(data2['SalStat'])

data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first=True)

#column names storing
columns_list = list(new_data.columns)

#separating inputs from the data
features = list(set(columns_list)-set(['SalStat']))

# Dependent variable Y, storing values in y
y=new_data['SalStat'].values

#independent varibales (input varibles ). storing the values from features
x=new_data[features].values


# Splitting the data into train and test data (train 70%, test 30%)

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3,random_state=0)


#Instance of logistic regression model

logistic = LogisticRegression()

# fitting the model  for the values x & y

logistic.fit(train_x,train_y)

#predict the model with test data
prediction = logistic.predict(test_x)
print(prediction)

# confusion matrix 
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

#accuracy score
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

# Misclassified values

(test_y != prediction).sum()


# if we build the model again with removing some insignificant variables, then some of the information may lose,
# the accuracy score may reduce depending on the insignificant variables.

# =====================================================

# KNN classifier model

#==================================================== 

#import KNN library
from sklearn.neighbors import KNeighborsClassifier

#import matplotlib for plots

import  matplotlib.pyplot as plt

# storing the k nearest neighbours classifier

KNN_classifier = KNeighborsClassifier(n_neighbors=5)

# Fitting the train x and y values for the model
KNN_classifier.fit(train_x,train_y)

# predicting the model with test x
prediction = KNN_classifier.predict(test_x)

#confisuin matrix checking

confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#accuracy score

accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

#miscalssified values
(test_y != prediction).sum()


#=================================
 # effect of k value on the classifier
 
Misclassified_sample = []
 # for k values between 1 to 20
 
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append(((test_y != pred_i).sum()))
                                 
                    
print(Misclassified_sample)




