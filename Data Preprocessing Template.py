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
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Missing Data Handle
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:])
X[:,1:]=imputer.transform(X[:,1:])"""

#Encoding categorical data
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""

#Test-Train Split of data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)