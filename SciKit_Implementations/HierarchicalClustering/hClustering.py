#implementing hierarchical clustering
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#Creating a dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Creating Heirarchy Clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'blue')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'green')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'cyan')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'magenta')
plt.title("Clusters")
plt.xlabel("Salary")
plt.ylabel("Spending")
plt.legend()
plt.show()
