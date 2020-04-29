#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:27:39 2020

@author: franklin velasquez fuentes

Neural Networks -> Classification problems
Convurotional -> Computer vision task

Thano -> numerical computations, it can run on GPU (Montreal)
Tensorflow -> for research, creating a lot of lines
Keras -> wraps Thano and Tensorflow, we can build Deep Learning models with few lines of code
"""

# PART 1 - Data Preprocessing

# From classification template

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


onehotencoder = OneHotEncoder(categorical_features=[1]) # only for countries it is necesary
X = onehotencoder.fit_transform(X).toarray()

"""
It is necessary to remove one column of contries, to avoid the dummy variable tramp !
There are 3 countries, we only need 2 columns to describe it
"""
X = X[:,1:]



# Splitting the Dataset into the Trainging Set and Test Set 
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set



# Feature Scaling - Importat, there is going to be a lot of computations, for easy calculations
# and to do not have any independent variable with more value than others.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)


# PART 2 - Now les make the ANN

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential #initialize ANN
from keras.layers import Dense #create layer



""""
Initialising an ANN options:
- Sequence of layers
- As a graph
""" 
classifier  = Sequential()


"""
Adding the input layer and the first hidden layers

    We have to remember the 7 steps of the Stochastic Gradient 
    
    Sigmoid function -> out put layer , is good for get probabilities
    Rectifier function -> for hidden layers
    
    We are going to create a ranking, with the probabilities of leaving the bank
    
    
    How many numbers of nodes? It is Art !
    * There are not rules, by experiment, it can be the avarage between the numbers of nodes of input and output
    * Artist: parameters tunning, cross validation technique, create another test set and experiment with
    diffferent numbres, and compare the performance.
    
    Dense(out_put_dim_parameters (nodes), how to initialize weights, activation function , input_dim (only for 1st hidden layer) )
"""

# Hidden Layer: we have 11 inputs, 1 output, avarage = 6 nodes. relu = rectifier function
# When we add the first hidden layer, we are also adding the input layer
classifier.add(Dense(output_dim=6,kernel_initializer="uniform",activation="relu",input_dim=11))


# Adding the second hidden layer
classifier.add(Dense(output_dim=6,kernel_initializer="uniform",activation="relu"))


# Adding the output layer
classifier.add(Dense(output_dim=1,kernel_initializer="uniform",activation="sigmoid"))

"""
If we have and output with more classes:
    change unit input, change activation, soft_maxt = sigmoid  funtion but appied to more categories
"""


# Compiling the ANN
"""
compile( algorithm to find the best weights, loss-cost-error funtion , type of metric to evaluate the model)

    adam = a stocastic gradient algorithm
    our lost funtions is not going to be the sum of squares, it is going to be the logaritmic loss 
    (base on sigmoid funtion), binary : binary_crossentropy , multi : categorical_crossentropy
    
    metrics = with this metrics the model is going to adjust its performance
"""

classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])


# Fitting the ANN to the Trainging Set
"""
chose the way of update weights , bach_size = number of observations to update
number of epoch, the number of rounds that the hole training set is going to pass trow the ANN

loss = 0.3370 , accuracy - 0.8582

My results:
loss: 0.4002 - accuracy: 0.8349
loss: 0.3342 - accuracy: 0.8631
"""

classifier.fit(X_train,Y_train,batch_size=10,epochs=100)



# PART 3 - Making the predictions and evaluating the model 


# Predicting the test set results
Y_pred = classifier.predict(X_test)


"""
Evaluating Performance of Predictions with Confusion Matrix
"""
from sklearn.metrics import confusion_matrix

"""
Y_pred have only provabilities, but Y_test have only 0-1, so we need to convert y_pred with a tresh hold
"""
# if Y_pred > 0.5, it returns true, else it returns false
Y_pred = (Y_pred > 0.5) 


# Params :
# y_true -> values in real life
# y_pred -> vector of predictions
cm = confusion_matrix(Y_test,Y_pred)

print((1507 + 216)/2000) # 0.8615  of acurracy in new preditions !!

"""
We can compare the acurracy of the confusion with the acurracy given by the metrics of the model
If they are the same we can validate our model

We can create a rank of the clients that are most likely to leave the bank
The company can take some decisions to fixh what they need
"""


"""
Without beeing an Artist, creating a creative ANN, we got a good acurracy.
But, if we modify the number of nodes, an change parameters, we can increase our acurracy
"""












#.

