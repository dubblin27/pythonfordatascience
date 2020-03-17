import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
loans =  pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv')
print(loans.info())
print(loans.describe())


#as purpose is an object type we have to create dummy numbers 
cat_feats = ['purpose']
finaldata= pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(finaldata.info())

x = finaldata.drop('not.fully.paid',axis=1)
y = finaldata['not.fully.paid']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
algo1 = DecisionTreeClassifier()
algo2 = RandomForestClassifier(n_estimators=200)
algo1.fit(x_train,y_train)
algo2.fit(x_train,y_train)
pred1 = algo1.predict(x_test)
pred2 = algo2.predict(x_test)

print(pred1 , "\n", pred2)
print("Decision Tree : ")
print(classification_report(y_test,pred1),"\n", confusion_matrix(y_test,pred1))
print("Random forest : ")
print(classification_report(y_test,pred2),"\n", confusion_matrix(y_test,pred2))
