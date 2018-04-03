#Implementing an Artificial Neural Network
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y=  dataset.iloc[:, 13].values

#Handling the categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_X1 = LabelEncoder()
X[:, 2] = encoder_X1.fit_transform(X[:, 2])
encoder_X2 = LabelEncoder()
X[:, 1] = encoder_X2.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Removing the Dummy Variable
X = X[:, 1:]

#Scaling the features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )
