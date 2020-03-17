import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
print(cancer.keys())
# print(cancer.info())
# print(cancer['DESCR']) 
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.info())
# print(df.info())
x = df 
y = cancer['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

algo = SVC()
algo.fit(x_train,y_train)
predictions = algo.predict(x_test)

print(confusion_matrix(y_test,predictions), "\n", classification_report(y_test,predictions))

param_grid = {
    'C' : [0.1,1,10,100,1000],
    'gamma': [1,0,.1,0.001,0.0001]
}
#cross-validation
grid = GridSearchCV(SVC(),param_grid,verbose = 3) #verbose - to get exact op
grid.fit(x_train,y_train)
print("\n",grid.best_params_) 
print("\n",grid.best_estimator_)

grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print("\n")
print(confusion_matrix(y_test,predictions), "\n", classification_report(y_test,predictions))

print(classification_report(y_test,grid_predictions))


