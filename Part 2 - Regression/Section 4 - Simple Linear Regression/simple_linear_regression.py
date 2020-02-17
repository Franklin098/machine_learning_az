# Simple Linear Regresion


# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Salary_Data.csv')
# independent variables
X = dataset.iloc[:, :-1].values  # Matrix of features [filas,columnas]  -> : = todas  -> :-1 = todas menos la ultima
# dependent variables matrix
y = dataset.iloc[:, 1].values


# Splitting the Dataset into the Trainging Set and Test Set --------------------------

from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test  = train_test_split(X, y, test_size = 1/3, random_state = 0 )  # test_size -> 20% on test set , 80% training set


# Fitting Simple Linea Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# Predicting the Test set result

# Creating a vector with the results of predictions
y_pred = regressor.predict(X_test)

# Visualising the Training set results. Predictios vs Reality


# Plotting training

plt.scatter(X_train,y_train, color = 'red' )
#Plotting regression line
plt.plot(X_train, regressor.predict(X_train) , color = 'blue' )
plt.title('Salary vs Experience (Training Set) ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plotting Test

plt.scatter(X_test,y_test, color = 'red' )
plt.scatter(X_test,y_pred, color = 'green' )
#Plotting regression line
plt.plot(X_train, regressor.predict(X_train) , color = 'blue' )
plt.title('Salary vs Experience (Test Set) ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()














