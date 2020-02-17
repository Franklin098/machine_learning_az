# Code by : Franklin098
#Decision Tree Regression

# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Position_Salaries.csv')

# X debe ser siempre considerada dentro de Python como una matriz, no como un vector, a:b b-> exclusivo
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values

"""
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set
"""

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit and then transform it 
X_test = sc_X.transform(X_test) # No es necesario hacer fit
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Fitting the Decision Tree Regression Model to the dataset

from sklearn.tree import DecisionTreeRegressor

#mse min square error
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predict a new result with Regression
y_pred = regressor.predict([[6.5]])


##Trap ! La gráfica no refleja el comportamiento de un Decision Tree !

# Visualising the Regression results
plt.scatter(X,y, color = 'red')
plt.plot(X , regressor.predict(X) , color = 'blue' )
plt.title("Truth or BLuff ( Regression  Model ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


## Esta gráfica si refleja el vardadero comportamiento de Decision Tree !

# Visualising the Decision Tree Regression results ( for higher resolution and smoother curve )
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid , regressor.predict(X_grid) , color = 'blue' ) # linea
plt.title("Truth or Bluff ( Decision Tree Regression ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



## Decision Trees are not interesting in 2D, but in more dimensions is realy powerful

