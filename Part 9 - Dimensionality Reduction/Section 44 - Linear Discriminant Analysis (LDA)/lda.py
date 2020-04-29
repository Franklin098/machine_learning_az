#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:42:55 2020

@author: franklinvelasquezfuentes

In PCA we are taking the variables that increase the most the variance
In LDA we are taking the variables that separate the most the classes of the dependent variabl
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
Here we  have to Apply LDA !

- After data preprocessing and 
- Before fitting to the model 
"""

from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA

# two linear discriminant, 2 new independent variables
lda = LDA(n_components=2)
# lda is supervised, so we neet the Y dependent variables.
X_train = lda.fit_transform(X_train,Y_train)
# we do not neet Y againt, because we already fit it before, now we only need to transform
X_test = lda.transform(X_test)




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
The 3 clases are correctly separated - 0 incorrect prediction

Each Wine is in the correct costumer region/segment of costumer. 

The wine business owner can be very confident, for each wine, based o its attributes he know 
to which of the 3 different segment costumer recomend it.

He knows to who recomend the new wines. It is very usefull to visualize it
"""


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
plt.xlabel("LD1")
plt.ylabel("LD2")
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
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()

plt.show()




from sklearn.svm import SVC


classifier = SVC(kernel="rbf",random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

cm2 = confusion_matrix(Y_test,Y_pred)



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
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()

plt.show()









