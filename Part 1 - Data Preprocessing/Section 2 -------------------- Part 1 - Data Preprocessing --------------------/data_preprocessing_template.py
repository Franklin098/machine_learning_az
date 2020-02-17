# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Data.csv')
# independent variables
X = dataset.iloc[:, :-1].values  # Matrix of features [filas,columnas]  -> : = todas  -> :-1 = todas menos la ultima
# dependent variables matrix
y = dataset.iloc[:, 3].values


# Splitting the Dataset into the Trainging Set and Test Set --------------------------

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set


# Feature Scaling  -----------------------------------
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit and then transform it 
X_test = sc_X.transform(X_test) # No es necesario hacer fit
"""





































