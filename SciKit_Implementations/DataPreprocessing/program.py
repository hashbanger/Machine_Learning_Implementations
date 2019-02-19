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

#Now we are going to encode the categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()  #encoding it as normal labels just for observation purpose
X[:,0] = labelencoder_X.fit_transform( X[:,0] )
onehotencoder= OneHotEncoder(categorical_features = [0]) #encoded the country columns
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()  #encoding it as normal labels just for observation purpose
y = labelencoder_y.fit_transform(y) #encoding the Yes as 1 and No as 0

#Now Splitting the dataset in test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

#Now scaling the features (Salary and Age columns)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
