import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data',index_col=0)

# scalar = StandardScaler()
# scalar.fit(df.drop('TARGET CLASS',axis = 1))
# scalar_features = scalar.transform(df.drop('TARGET CLASS',axis = 1))

# df_feat = pd.DataFrame(scalar_features,columns=df.columns[:-1])
# # print(df_feat.head())

# x = df_feat 
# y = df['TARGET CLASS']

# x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
# knn = KNeighborsClassifier(n_neighbors=300)
# knn.fit(x_train,y_train)
# pred = knn.predict(x_test)
# print(pred)

# print(confusion_matrix(pred,y_test),classification_report(pred,y_test))

from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data',index_col=0)
dp = df.drop('TARGET CLASS',axis=1)

ss = StandardScaler()
ss.fit(dp)

Scaler_features = ss.transform(dp)

df_feat = pd.DataFrame(Scaler_features,columns=df.columns[:-1])

x= df_feat 
y = df['TARGET CLASS'] 
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
algo=KNeighborsClassifier(n_neighbors=37)
algo.fit(X_train,y_train)
pred = algo.predict(X_test)
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


error = []
for i in range(1,100):
    algo = KNeighborsClassifier(n_neighbors=i)
    algo.fit(X_train,y_train)
    pred_i = algo.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
print(np.max(error))

plt.plot(range(1,100),error,marker='o')
plt.show()