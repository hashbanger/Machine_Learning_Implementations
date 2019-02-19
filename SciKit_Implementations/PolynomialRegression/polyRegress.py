#Building Rgression model with polynomial features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[: ,1:2].values #To convert it from vector to matrix
y = dataset.iloc[: ,2].values

#Now fitting the regression model the without polyfeatures
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Now constructing and fiting with polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#now visualising the simple regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Level vs Salary Prediction")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

#now visualising the poly regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') #Don't put X_poly for generalization
plt.title("Level vs Salary Prediction")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

#Now predicting a single value for any employee by Simple Regression Model
print(lin_reg.predict(6.5))
#Now predicting a single value for any employee by Polynomial Regression Model
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))
