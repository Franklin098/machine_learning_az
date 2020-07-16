#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:53:03 2020

@author: franklinvelasquezfuentes

XG Boost

Taking again the problem of Churn_Modelling, in wich based on some features we are trying
to classify if the customer is going to leave the bank or not.

We did it using ANN, but it took a lot of epocs and time. Now applying XG Boost we are going
to get the same accuracy (86%), but it is going to be executed so much faster.

XG Boost is one of the best models in terms of permorfance and speed

In XG Boost it is not necessary feature skealing, so we still have the same labels, we
keep the original interpretation of or model.

1) High Performance  2) Fast execution speed 3) Keep original interpretation

"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as  pd


dataset = pd.read_csv('Churn_Modelling.csv')

# Only rows 3-12 are relevant to predict
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



"""
We have categorical variables ! we need to encode them
categorical variables:  country: (france, spain, germany) gender: (male,female)
onehotencoder -> to create new columns, create dummy variables
"""

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) # encoding countries
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2]) # encoding gender

from sklearn.compose import  ColumnTransformer

column_transformer = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(column_transformer.fit_transform(X),dtype=np.float)


"""
It is necessary to remove one column of contries, to avoid the dummy variable tramp !
There are 3 countries, we only need 2 columns to describe it
"""
X = X[:,1:]


# Splitting the Dataset into the Trainging Set and Test Set 
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set




# Fitting XGBoost to the Training Set
"""
XG Boost uses trees, so n_estimators is the numbers of trees
"""
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,Y_train)


# Predicting the test set results
Y_pred = classifier.predict(X_test)

"""
Evaluating Performance of Predictions with Confusion Matrix
"""
from sklearn.metrics import confusion_matrix

"""
Y_pred have only provabilities, but Y_test have only 0-1, so we need to convert y_pred with a tresh hold
"""
# Params :
# y_true -> values in real life
# y_pred -> vector of predictions
cm = confusion_matrix(Y_test,Y_pred)



"""
Applying K-Fold Cross Validation
"""
from sklearn.model_selection import cross_val_score
"""
It is going to return the 10 accuracies
n_jobs = -1, when working to large datasets
"""
accuracies = cross_val_score(estimator=classifier,X = X_train, y= Y_train,cv=10)
accuracies.mean() # 
accuracies.std()


"""
Accuracy = 86.45%% 

Accurracy applying K-Fold Cross Validation =  86.25%
Variance = 1.017 %
"""


















































