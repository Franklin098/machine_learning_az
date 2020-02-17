# Naive Bayes


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



# Fitting classif¡er Regression
# Create your classifier here ! :


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
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

plt.title("Naive Bayes (Training Set)")
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

plt.title("Naive Bayes (Test Set)")
plt.xlabel("X1 - Edad")
plt.ylabel("X2 - Sueldo Estimado")
plt.legend()

plt.show()






