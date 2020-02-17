# Ejemplo de Multiple Linea Regression


# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('50_Startups.csv')
# independent variables
X = dataset.iloc[:, :-1].values  # Matrix of features [filas,columnas]  -> : = todas  -> :-1 = todas menos la ultima
# dependent variables matrix
y = dataset.iloc[:, 4].values





from sklearn.preprocessing import LabelEncoder , OneHotEncoder

# Encoding categorical data

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 

# Dummy encoding to do not set a value to categorical data

onehotencoder = OneHotEncoder(categorical_features=[3]) # parametro -> index of column
X = onehotencoder.fit_transform(X).toarray()



###  New :  Avoiding the Dummy Varable Trap
# Quitando la columna 0, porque ? -> Recordar que para dummy variables es n-1
X = X[:,1:]   
# La libreria de regresion lineal lo hace solo, pero tambien podemos agregarlo


# Splitting the Dataset into the Trainging Set and Test Set --------------------------

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  # test_size -> 20% on test set , 80% training set




# Multiple Lineal Regression Library

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,Y_train)

# We are not going to plot becase of multiple dimmensions

# Predicting the Test set results

y_pred = regressor.predict(X_test)




# Building optimal model using Backward Elimination -----------------------
# Find the most significant variables in the model


# Preprocessing for Backward Elimination

# Adding a column of 1s because   "b0"  term  ->  b0x0  with x0 = 1 
# this library do not takes in count b0 therm, regressor yes

X = np.append( arr =  np.ones((50,1)).astype(int)  ,  values =  X , axis = 1 )


import statsmodels.formula.api as sm


# Create new matrix of features optimal, the ones that have high impact at the profict

X_opt  = X[:, [0,1,2,3,4,5] ]  # inicializa como la original

# Review backward elimination algorithm at powerPoint presentation
# significance level = 0.05

#Ordinary Least Squares
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()


# predictor  -> independent variable. Aqui hacer el bucle
# Summary function -> return statistical matrics of the model (including p-value)

regressor_OLS.summary()
#  in base of the results we need to eliminate X2


# ---- Repeating  ----- 

X_opt  = X[:, [0,1,3,4,5] ] 
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()


# ---- Repeating  ----- 

X_opt  = X[:, [0,3,4,5] ] 
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()


# ---- Repeating  ----- 

X_opt  = X[:, [0,3,5] ] 
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()


# ---- Repeating  ----- 

X_opt  = X[:, [0,3] ] 
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()





# automathic


import statsmodels.formula.api as sm

def backwardElimination(x, sl):
    
    numVars = len(x[0])
    
    for i in range(0,numVars):
        
        regressor_OLS = sm.OLS( endog = y , exog = x).fit()
        pvalues = regressor_OLS.pvalues
        maxVar = max(pvalues).astype(float)
        
        if maxVar > sl :
            
            for j in range(0, numVars - i) :
                
                if(pvalues[j].astype(float) == maxVar):
                    x = np.delete(x,j,1)
                    
    regressor_OLS.summary()
    return x



sl = 0.05
X_opt = X[:, [0,1,2,3,4,5] ]
X_modeled = backwardElimination(X_opt,sl)

        






























