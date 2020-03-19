# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:45:42 2018

@author: ASH
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#Fitting Decision Tree Regression model to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting a new result with Polynomial Regression
y_pred=regressor.predict(6.5)


#Visualising the polynomial Regressiion results()for higher resolution and smoother curves)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff[Decision Tree Regression Model]')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
