#performing Multilinear Regression optimally
#Using Backward Elimination
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[ :, :-1].values
y = dataset.iloc[ :,4].values

#Encoding the categorical variable column states
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Excluding the dummy variable manually just in case
#python libraries don't need it done explicitly though

X = X[:, 1:]

#Splitting the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Now fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Now predicting the values for X_test
y_pred = regressor.predict(X_test)

#Bulilding optimal model
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 )
#To append at the end do below:
#X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1 )
X_opt = X[: ,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #To observe P values
#Removing highest P value column i.e. 2
X_opt = X[: ,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: ,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: ,[0, 1, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: ,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: ,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#The final features are R&D Spend and the bias

#Splitting the new model
from sklearn.cross_validation import train_test_split
X_opttrain, X_opttest, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

#Now fitting the optimized model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_opttrain, y_train)

#Now predicting the values for new X_test
y_pred2 = regressor.predict(X_opttest)




