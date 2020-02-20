import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# print(train.describe())

# sns.boxplot(x='Pclass',y='Age',data=train)
# plt.show()

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

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
# plt.show() #missing info of cabin

train.drop('Cabin',axis=1,inplace = True)#as cabin has lot of missing values

train.dropna(inplace = True) # to deal with missing values

#to place in dummy values
#convert catogorical variables into dummies to replace into 0s and 1s
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# print(embark.head()) 

train = pd.concat([train,sex,embark],axis=1)

# print(train.head(2))
# print(train.head(2))
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis = 1, inplace = True)

# print(train.head(2))
# print(train.tail(2))

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train , X_test , y_train, y_test = train_test_split(X,y, test_size = 0.33,random_state=101)
logmodel = LogisticRegression(max_iter=1200000)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("predictions - survived : " , predictions)
#shows the report
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

# def binaryToDecimal(n): 
#     return str(int(n,2)) 
  
  
 
# if __name__ == '__main__': 
#     print(binaryToDecimal(predictions)) 

