#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:01:29 2020

@author: franklinvelasquezfuentes

K_fold_Cross = evaluate model performance
Grid_search = improve model performance

Grid Search is used to build more powerful models, finding the optimal values of hyperparameters,
those parameters that we choose, like the kernel, those parameteres that are not learned.


What is the optimal value of the hyperparemeters?
- Grid Search is going to find it.

How to know what model to choose for my business problem?

1. Only looking to dependent variable we can know if it is regression (continious), 
classification (categorical), or if there are not dependent variables it is a clustering problem.

2. Does the problem is linear? We need to analyse the data, see if it is linearly separable
 and based on that choose:
    linear: SVM (classification), 
    not linear: kernel SVM (classification)
    
Grid Search is going to help us to know if it is linear or not
"""


"""
Using the problem of Social Network Ads, based on the salary and age, if the people
are going to buy or not the car.

Grid Search , to decide if use SVM or Kernel SVM (gama, penalty parameter)
"""

# Support Vector Machine - SVM
# Data Preprocessing
import numpy as np  # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets

dataset = pd.read_csv('Social_Network_Ads.csv')

# independent variables
X = dataset.iloc[:, [2,3]].values  # Matrix of features [filas,columnas]  -> : = todas  -> :-1 = todas menos la ultima
# dependent variables matrix
y = dataset.iloc[:, 4].values


# Splitting the Dataset into the Trainging Set and Test Set --------------------------

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.25, random_state = 0 )  # test_size -> 20% on test set , 80% training set


# Feature Scaling  -----------------------------------
# Para poner todo en una misma escala

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit and then transform it 
X_test = sc_X.transform(X_test) # No es necesario hacer fit



# Fitting classifi.¡er Regression
# Create your classifier here ! :


from sklearn.svm import SVC


classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting the test set results

Y_pred = classifier.predict(X_test)

"""
Evaluating Performance of Predictions with Confusion Matrix
"""

from sklearn.metrics import confusion_matrix

# Params :
# y_true -> values in real life
# y_pred -> vector of predictions
cm = confusion_matrix(Y_test,Y_pred)



"""
Applying K-Fold Cross Validation - Evaluating Performance
"""

from sklearn.model_selection import cross_val_score

"""
It is going to return the 10 accuracies
n_jobs = -1, when working to large datasets
"""

accuracies = cross_val_score(estimator=classifier,X = X_train, y= Y_train,cv=10)
accuracies.mean()

"""
mean = 0.90333 
This 90% acurracy is the relevant evaluation of our model, not one of the acurracies,
is the mean of at least 10 evaluations
"""
accuracies.std()
"""
Standard Desviation
 0.06574360974438671
6% of desviation is low, it is okay, most of the time we do not have to much variance.

We are at Low Bias and Low variance category ! 
"""




"""
Applying Grid Search - To find the best model and the best parameters ! - Improve Model
"""

from sklearn.model_selection import GridSearchCV

"""
A list of dictionaries, for each  dictionary : key =  parameter we want to optimize,
for each key we are going to give it values to evaluate and to find the best one

SVM , our model have a lot of hyperparameters, we can see them in Help area, looking at the
parameters that the model receives like : 
    C (penalty to prevent overfitting), kernel, gamma(default 1/n)
"""

parameters = [ { 'C': [1,10,100,1000] , 'kernel':['linear']},
              { 'C': [1,10,100,1000] , 'kernel':['rbf'] , 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] }
    ]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1
                           )

"""
In the first line item of our parameters variable we evaluate the linear option
In the second line of our parameters variable we evaluate the nonlinear option

cv = 10 , 10 cross validation, like k_fold_cross_validation
n_jobs to get all the power available
"""

grid_search = grid_search.fit(X_train, Y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

"""
Results:
best_accuracy = 0.90666

best_params:
    
    C = 1
    gamma = 0.7
    kernel = rbf
"""



# Graficando los Resultados


from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_train, Y_train


X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green'))(i), label = j )

plt.title("SVM (Training Set)")
plt.xlabel("X1 - Edad")
plt.ylabel("X2 - Sueldo Estimado")
plt.legend()

plt.show()




from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_test, Y_test


X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green'))(i), label = j )

plt.title("SVM (Test Set)")
plt.xlabel("X1 - Edad")
plt.ylabel("X2 - Sueldo Estimado")
plt.legend()

plt.show()


































