# Code by : Franklin098

# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Position_Salaries.csv')

# X debe ser siempre considerada dentro de Python como una matriz, no como un vector, a:b b-> exclusivo
X = dataset.iloc[:, 1:2].values 

y = dataset.iloc[:, 2].values

# Solo son 10 datos , no vale la pena hacer training set and test set.
"""
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set
"""

# La libreria hace solo el Feature Scaling, no necesario.

# Fitting Lineal Regression to dataset

from sklearn.linear_model import  LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures( degree = 4 )

# Transforma la matrix X  lineal , agrega X^2 , X^3 ... etc
# ahora en X_poly tenemos 3 columnas :  b , X , X^2
X_poly = poly_reg.fit_transform(X) 


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


# Visualising the Linear Regression results


plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue' )
plt.title("Truth or BLuff ( Linear Regression ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Visualising the Polynomial Regression results


# Agregando mas pasos o puntos para obtener mayor suabidad en la curva

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)


plt.scatter(X,y, color = 'red')
plt.scatter(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')

# No basta con simplemente remplazar por lin_req_2.predict(X) es X_poly, también en el primer parametro 
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue' )

plt.title("Truth or Bluff ( Polynomial Regression ) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Para ir mejorando la predicion subimos el grado del polinmio

# Predict a new result with linear regression

lin_reg.predict([[6.5]])


# Predict a new result with polynomial regresion

# OJO : no es solo meter el valor numerico a predict ! ( por X_poly)

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

## Truth ! 

reg_label = "Inliers coef:%s - b:%0.2f" % \
            (np.array2string(lin_reg_2.coef_,
                             formatter={'float_kind': lambda fk: "%.3f" % fk}),
            lin_reg_2.intercept_)
print(reg_label)




intercept, coefficients = lin_reg_2.intercept_, lin_reg_2.coef_


print('intercept:', intercept)


print('coefficients:', coefficients, sep='\n')
