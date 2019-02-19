#Making predictions using a Decision Tree Regression Model
#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Getting dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Creating the regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#predicting the salary for employee at 6.5 level]
y_pred = regressor.predict(6.5)

#Visualising the Predictions
X_grid = np.arange(min(X), max(X), 0.001 )
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
