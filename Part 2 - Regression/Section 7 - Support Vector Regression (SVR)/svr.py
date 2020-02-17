# Code by : Franklin098
# SVR


# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Position_Salaries.csv')

# X debe ser siempre considerada dentro de Python como una matriz, no como un vector, a:b b-> exclusivo
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2:3].values

"""
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set
"""

# Feature Scaling  --- SVR package does not includes Feature Scaling !

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR Model to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')  # rbf -> para modelos no lineales
regressor.fit(X,y)


# Predict a new result with Regression
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)

# No podemos meter solamente 6.5 hay que transformarlo a feature scaling
# Despues hay que invertir la escala para obtener el valor real

# Visualising the SVR results
plt.scatter(X , y, color = 'red')
plt.plot(X , regressor.predict(X) , color = 'blue' )
plt.title("Truth or BLuff ( SVR  ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Visualising the Regression results ( for higher resolution and smoother curve )
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid , regressor.predict(X_grid) , color = 'blue' ) # linea
plt.title("Truth or BLuff ( Regression  Model ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()








