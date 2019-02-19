#Preprocessing the data

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #choosing the independent variable
y = dataset.iloc[:,3].values #choosing the dependent variable
#if you drop .values then place .iloc everywhere X.iloc[:,1:3]


#Now Splitting the dataset in test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

#Now scaling the features (Salary and Age columns)
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''