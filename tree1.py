import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
df =pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv')
# print(df.head())
# print(df.info())
# print(df.describe)
# sns.pairplot(df,hue = 'Kyphosis')
# plt.show()
print(df['Kyphosis'].value_counts())
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.3) 

algo = DecisionTreeClassifier()
algo.fit(X_train,y_train)
pred = algo.predict(X_test)
print(pred)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


algo1 = RandomForestClassifier(n_estimators=200)
algo1.fit(X_train,y_train)
pred1 = algo1.predict(X_test)
print(pred1)
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))