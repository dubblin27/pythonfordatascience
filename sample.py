import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report 

df = pd.read_csv('dataset and ipynb/17-K-Means-Clustering/College_Data',index_col=0)
# print(df.head())
# print(df.info())
# sns.lmplot(x='Room.Board', y = 'Grad.Rate',data=df,hue='Private',fit_reg=False)
# plt.show()

# print()
# for i in df:
#     if df[df['Grad.Rate']>100] :
        

# df['Grad.Rate'].replace(to_replace = df['Grad.Rate']>100 , value =100) 
# print(df[df['Grad.Rate'] > 100])
# # print(df['Grad.Rate'] > 100)
# # print("hi")
# # print(df[df['Grad.Rate']>100])
df[df['Grad.Rate'] > 100] = 100

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0 

df['Cluster'] = df['Private'].apply(converter)
print(df[['Cluster','Private']])

print(kmeans.labels_)
print(confusion_matrix(df['Cluster'],kmeans.labels_),"\n",classification_report(df['Cluster'],kmeans.labels_))