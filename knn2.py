import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data',index_col=0)

scalar = StandardScaler()
scalar.fit(df.drop('TARGET CLASS',axis = 1))
scalar_features = scalar.transform(df.drop('TARGET CLASS',axis = 1))

df_feat = pd.DataFrame(scalar_features,columns=df.columns[:-1])
# print(df_feat.head())

x = df_feat 
y = df['TARGET CLASS']

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
knn = KNeighborsClassifier(n_neighbors=300)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print(pred)

print(confusion_matrix(pred,y_test),classification_report(pred,y_test))