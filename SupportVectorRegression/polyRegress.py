#Creating a Support Vector Regression Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing datasets
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Now Feature scaling as our SVR class doesn't do it by itself
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.reshape(y, (len(y),1))
y = sc_y.fit_transform(y) #asks for a matrix rather than vector

#Creating SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #Using the Gaussian Kernel
regressor.fit(X, y)

#Now predicting the values
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#Now  increassing the resolution of the graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
