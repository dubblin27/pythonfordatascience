import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')
print(train.head())

# sns.heatmap(train.isnull())
# plt.show()

# sns.countplot(x='Survived',hue='Pclass', data=train)
# plt.show()

# sns.distplot(train['Age'].dropna(),kde=False,bins=30)
# plt.show()

# train['Age'].plot.hist()
# plt.show()
print(train.info())
print(train.describe())

sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()

#avg age of people travelling in 1st, 2nd , 3rd class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37 
        elif Pclass ==2 :
            return 29 
        else :
            return 24 
    else :
        return Age 

train['Age'] = train[['Age', 'Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
plt.show() #missing info of cabin

train.drop('Cabin',axis=1,inplace = True)#as cabin has lot of missing values

train.dropna(inplace = True)

