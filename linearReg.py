from sklearn.model_selection import train_test_split 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('/media/sabrish2701/Sabrish1TB/Coding/pythonfordatascience/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')
# print(df.head())

# print(df.info())

# print(df.describe())

# print(df.columns)

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

X_train,X_test, y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
print(lm.coef_)
print(pd.DataFrame(lm.coef_,x.columns,columns=['Coeff']))
predictions = lm.predict(X_test)
print(predictions)

plt.scatter(y_test,predictions)
# plt.show()


# sns.distplot((y_test - predictions))
# plt.show()

mae= metrics.mean_absolute_error(y_test,predictions)
print(mae)
maes = metrics.mean_squared_error(y_test, predictions)
print(maes)
rmars = np.sqrt(maes)
print(int(rmars))