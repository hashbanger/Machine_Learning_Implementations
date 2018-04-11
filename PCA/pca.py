#implementing Principal Component Analysis for dimensionality reduction
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
