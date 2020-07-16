#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:50:44 2020

@author: franklinvelasquezfuentes

Kernel PCA

PCA and LDA works with linear problems, when the data is linearly separable

Kernel PCA is used when data is not linear separable, it is a kind of special PCA,
it takes the principal components.

Use this technique when PCA is not working well, because the problem is not linear
"""


# Logistic Regresion


# Data Preprocessing

import numpy as np  # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


"""
Estamos tratando de predecir cuáles personas en esta red social tienen
más probabilidades de comprar un nuevo SUV  en base al data set

Tomaremos la edad y el salario estimado como variables independientes columnas [2,3]
"""


dataset = pd.read_csv('Social_Network_Ads.csv')

"""
Si usamos el modelo de regresión logística, vemos que el problema no es linea, las predicciones son malas,
necesitamos separar las clases por medio de una curva, no por una linea. No
es linealmente separable

En vez de utilizar otro modelo como SVM, Decision Tree, Naybe-Bayes, vamos a seguir utilizando
el modelo de Regresión Lógica lineal, pero vamos a aplicar Kernel PCA. 
Una línea continuará siendo nuestro divisor, pero vamos a aplicar el Kernel PCA para
mapear la data en una dimensión superior y luego utilizar PCA para reducir las dimensiones
y obtener 2 nuevas variables que explican la mayor varianza, y estas si serán separables por 
una línea
"""



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


"""
Applying Kernel PCA
"""
from sklearn.decomposition import  KernelPCA 

# Gaussian Kernel = rbf
kpca = KernelPCA(n_components=2,kernel="rbf")
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

"""
Kernel will increase the data to a higher dimension where it is linearly sepparable, 
then with pca it will decrease to a fewer dimension, where we can use the logistic
regression to separate the data.
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

"""
"""


# Graficando los Resultados


from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

X_set , Y_set = X_train, Y_train

"""
Para graficarlo se utiliza el modelo para predecir un monton de pixeles
np.arange() regresa una lista de puntos que van de un min a un max con un step

X1 -> edad
X2 -> sueldo estimado
"""

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))


plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75 , cmap = ListedColormap(('red','green')))


plt.xlim(X1.min(),X1.max())
plt.xlim(X2.min(),X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set== j,1],
                c = ListedColormap(('red', 'green'))(i), label = j )

plt.title("Logistic Regression (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()






