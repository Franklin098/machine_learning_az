#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:35:55 2020

@author: franklinvelasquezfuentes

We are working for a business owner, our independent variables are features of wine like
alcohol, ash, magnesium, profine, etc. The owner already did a clustering process to classify
in 3 types his costumers, depending on the info of the wine. Each type of customres corresponds
to 3 types of wine.

The logist regression is going to try to predict to which type of costumer applies one
wine based on its attributes

Then we want to visualize if the predictions are in the correct spot, there are 13 dimentions, so
we need to apply some dimensional reduction techniques. PCA
"""

# Logistic Regression

# Data Preprocessing
import numpy as np  # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets



dataset = pd.read_csv('Wine.csv')


# independent variables
X = dataset.iloc[:, 0:13].values 
# dependent variables matrix
y = dataset.iloc[:, 13].values


# Splitting the Dataset into the Trainging Set and Test Set --------------------------
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set


# Feature Scaling  -----------------------------------
# Para poner todo en una misma escala
"""
With PCA and LDA we need to apply Feature Scaling !!
"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit and then transform it 
X_test = sc_X.transform(X_test) # No es necesario hacer fit


"""
Here we  have to Apply PCA !

- After data preprocessing and 
- Before fitting to the model 
"""

from sklearn.decomposition import  PCA


"""
First we need to check the vector of Variances, to look at the highest variances.
"""
#pca = PCA(n_components=None)
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

"""
The sum of the explained_variance vector is 1. 
If we take the first 2 values : 0.369 + 0.193  = 0.56, 
they will explain the 56% of the variance

So, now we will take those 2  principal components that explain the most the variance
Now, we restart the Kernell, and change the value of n_components from None to 2.

After that if we look to X_train and X_test they only have 2 variables
"""




"""
Fitting Logistic Regression
"""

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,Y_train)

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


# Graficando los Resultados


from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_train, Y_train

"""
Para graficarlo se utiliza el modelo para predecir un monton de pixeles
np.arange() regresa una lista de puntos que van de un min a un max con un step
"""

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green','blue')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j )

plt.title("Logistic Regression (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()






from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_test, Y_test


X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green','blue')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j )

plt.title("Logistic Regression (Test Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()




from sklearn.svm import SVC


classifier = SVC(kernel="rbf",random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)



from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_train, Y_train

"""
Para graficarlo se utiliza el modelo para predecir un monton de pixeles
np.arange() regresa una lista de puntos que van de un min a un max con un step
"""

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green','blue')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j )

plt.title("Logistic Regression (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()








