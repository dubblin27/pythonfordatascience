import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GrifSearchCV
cancer = load_breast_cancer()
(cancer.keys())
# print(cancer.info())
# print(cancer['DESCR']) 
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
# print(df.info())
x = df 
y = cancer['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

algo = SVC()
algo.fit(x_train,y_train)
predictions = algo.predict(x_test)

print(confusion_matrix(y_test,predictions), "\n", classification_report(y_test,predictions))

param_grid = {}

