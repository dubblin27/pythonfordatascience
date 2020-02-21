import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv')
print(train.head())
print(train.columns)
print(train.describe())
# train['Age'].hist(bins=30)
# plt.show()

# sns.jointplot(x='Age',y='Area Income', data=train)
# plt.show()

X = train[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y = train['Clicked on Ad']
reg = LogisticRegression(max_iter=1200000)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state = 101)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))