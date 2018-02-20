#!/usr/bin/env python3
#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv') 
X = dataset.iloc[:, :-1].values #Use 2D array even for a single column to avoid some errors.
y = dataset.iloc[: , -1].values

#Splitting the data to 1/3rd
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1/3 )

#fitting the regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Now predicting
y_pred = regressor.predict(X_test)

#Now Visualising
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()
