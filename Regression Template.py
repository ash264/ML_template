# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:36:54 2018

@author: ASH
"""
#Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
#Avoiding the dummy variable trap
X=X[:,1:]"""

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Regression model to the Dataset
#Create your own regressor



#Predicting a new result with Polynomial Regression
y_pred=regressor.predict(6.5)

#Visualising the polynomial Regressiion results
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.title('Truth or Bluff[Regression Model]')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial Regressiion results()for higher resolution and smoother curves)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff[Regression Model]')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



