# Data Preprocessing
# Importing the libraries

import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


# Importing the dataset -> using pandas

dataset = pd.read_csv('Data.csv')
# independent variables
X = dataset.iloc[:, :-1].values  # Matrix of features [filas,columnas]  -> : = todas  -> :-1 = todas menos la ultima

# dependent variables matrix

y = dataset.iloc[:, 3].values

# Handle missing data -> Replace null by the mean/average of the other values.

from sklearn.impute import SimpleImputer  


imputer = SimpleImputer(missing_values= np.nan,strategy= 'mean') # axis = 0 -> mean of columns / axis = 1 -> mean of rows
imputer.fit(X[:, 1:3])  # 1:2 -> [num,num) el numero de abajo es inclusivo, el de arriba exclusivo
 
X[:, 1:3] = imputer.transform(X[:,1:3])  # reemplazando valores originales con modificados






# Categorical Data -> Country and Purchased : because they contains categories
# Countries -> France, Spain, Germany 
# We only want number on equations, not strings -> we have to encode it

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) # Encoding Countries

'''
X[:,0] = labelencoder_X.fit_transform(X[:,0]) -> Aqui hay un problema

Spain = 0 / Germany = 3 no tiene sentido que los países tengan un valor numérico ya que no es algo medible,
en la ecuación se toma ese valor. Para corregir esto usamos Dummy Encoding. Para cada categoría se hace una columna
'''

onehotencoder = OneHotEncoder(categorical_features=[0]) # parametro -> index of column
X = onehotencoder.fit_transform(X).toarray()

# En Y no hace falta usar Dummy Encoding porque es variable dependiente y el algoritmo ya lo toma como categoria

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the Dataset into the Trainging Set and Test Set --------------------------

"""
Machine that is going to lear something, and algorithm or a model is learning.
Trainging Set -> We build the machine learning model
Test Set -> We set the performance of the machine model

Both are diferent. With test we test that the model can adapt to new situations
"""


from sklearn.model_selection import train_test_split



X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set


# Feature Scaling  -----------------------------------

"""
Cuando 2 variables no tienen la misma escala puede causar problema en los modelos
debido al Euclidean Distance (distancia entre 2 puntos)
La distancia será dominada por la columna que tenga mayor distancia
Neceistamos poner las variables en la misma escala y rango

Metodos para Feature Scaling y ecuaciones -> 1) Standardistaion  2) Normalisation
"""

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit and then transform it 
X_test = sc_X.transform(X_test) # No es necesario hacer fit

# Do we need to fit and transform the dummy variables ? 
# Whit feature scalint the algorithms are faster

# y is a categorical variable only 1 to 0  is not necesary feature scaling

# Some libraries automatically apply feature scaling others no







































