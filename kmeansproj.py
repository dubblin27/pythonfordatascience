import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('dataset and ipynb/17-K-Means-Clustering/College_Data',index_col=0)
print(df.head())

print(df.info())

##alternative for pylpot-scatterplot
# sns.lmplot(x='Room.Board', y = 'Grad.Rate',data=df,hue='Private')
# plt.show()
# sns.lmplot(x='Room.Board', y = 'Grad.Rate',data=df,hue='Private',fit_reg=False)
# plt.show()

# print(df[df['Grad.Rate']>90])
df['Grad.Rate']['Cazenovia College'] = 100
# print(df[df['Grad.Rate']>100])

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
# print(kmeans.cluster_centers_)
def converter(private):
    if private == 'Yes':
        return 1 
    else:
        return 0 
df['Cluster'] = df['Private'].apply(converter)
print(df)

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print("\n")

print(classification_report(df['Cluster'],kmeans.labels_))