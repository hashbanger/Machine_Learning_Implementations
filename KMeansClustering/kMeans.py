#implementing the K-Means CLustering Algorithm
#impoting the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#To find the optimal no. of clusters using elbow visualization
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) 
plt.plot(range(1,11),wcss)
plt.xticks(np.arange(1,11))
plt.yticks(np.arange(1,max(wcss),10000))
plt.grid()
plt.title('Within Clusters Squared Sum Visualisation')
plt.xlabel('No. of CLusters')
plt.ylabel('WCSS Sum')
plt.show()

#applying the K-Means using optimal no. of clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init =10)
y_kmeans = kmeans.fit_predict(X)
