import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df =pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv')
# print(df.head())
# print(df.info())
# print(df.describe)
sns.pairplot(df,hue = 'Kyphosis')
plt.show()