#performing Multilinear Regression
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

#Now predicting the values for y_test
y_pred = regressor.predict(X_test)
