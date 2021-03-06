import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data',index_col=0)
print(df.head())

scaler =  StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis = 1))
scaler_features = scaler.transform(df.drop('TARGET CLASS',axis = 1)) #peforms standarization
print(scaler_features)

df_feat = pd.DataFrame(scaler_features,columns=df.columns[:-1])
print(df_feat.head())

X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 101)

knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#to find the best K value
# error_rate = []
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))

# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
# plt.title('Error vs K')
# plt.xlabel('K')
# plt.ylabel('Error')
# plt.show()