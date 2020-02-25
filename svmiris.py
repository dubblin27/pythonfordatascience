#@author: sabrish
import numpy as np 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV



iris = sns.load_dataset('iris')
print(iris.keys())
sns.pairplot(iris,hue='species', palette='Dark2')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade =True,shade_lowest=False)
plt.show()
x = iris.drop('species',axis=1)
y= iris['species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 101)
algo = SVC()
algo.fit(x_train,y_train)
svc_predictions = algo.predict(x_test)
print("\n")
# print("confusion_matrix : \n",confusion_matrix(y_test,svc_predictions),"\n","classification_report: \n",classification_report(y_test,svc_predictions))


#using Grid Search 

param_grid = {
    'C' : [0.1,1,10,100,1000],
    'gamma': [1,0,.1,0.001,0.0001]
}

algo1 = GridSearchCV(SVC(),param_grid,verbose = 3)
algo1.fit(x_train,y_train)
print("\n",algo1.best_params_) 
print("\n",algo1.best_estimator_)

grid_predictions = algo1.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print("\n")
#normal predictions: 
print("confusion_matrix : \n",confusion_matrix(y_test,svc_predictions),"\n","classification_report: \n",classification_report(y_test,svc_predictions))

#grid predictions
print("confusion_matrix : \n",confusion_matrix(y_test,grid_predictions),"\n","classification_report: \n",classification_report(y_test,grid_predictions))
