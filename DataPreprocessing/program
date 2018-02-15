#Preprocessing the data

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values #choosing the independent variable
y=dataset.iloc[:,3].values #choosing the dependent variable
#if you drop .values then place .iloc everywhere X.iloc[:,1:3]

#importing sci-kit learn's preprocessing lib
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)#handling Nan to mean along columns
imputer = imputer.fit(X[:, 1:3])
X[:,1:3]=imputer.transform(X[:, 1:3])
