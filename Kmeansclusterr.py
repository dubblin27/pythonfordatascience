import seaborn as sns 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans


#to use atrificial data 

from sklearn.datasets import make_blobs 
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101,shuffle= True)

# n_samples : int, optional (default=100) The total number of points equally divided among clusters.
# n_features : int, optional (default=2) The number of features for each sample.
# centers : int or array of shape [n_centers, n_features], optional (default=3) The number of centers to generate, or the fixed center locations
# cluster_std: float or sequence of floats, optional (default=1.0) The standard deviation of the clusters



# print(data[1])
# plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
# plt.show()


kmeans = KMeans(n_clusters=5)
kmeans.fit(data[0])

# print(kmeans.cluster_centers_)

# print(kmeans.labels_)

fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))
#sharey to align the horizontal or vertical axis

ax1.set_title('K_means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

ax2.set_title('original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()