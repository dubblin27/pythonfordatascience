import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])

print("2 main targets : ",*cancer['target_names'])
# to scale a data to check if data has single unit Variance 

scaler = StandardScaler()
scaler.fit(df) 
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data = scaler.transform(df)

#PCA 
pca = PCA(n_components = 2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First Principle Component')
plt.ylabel('Second PC')
plt.show()
# print(pca.components_)
df_comp = pd.DataFrame(pca.components_,columns = cancer['feature_names'])
print(df_comp)
sns.heatmap(df_comp,cmap='plasma')
plt.show()