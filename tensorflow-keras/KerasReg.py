#predict a price of a house 
# for king county -seattle USA
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score

df = pd.read_csv('TensorFlow_FILES/DATA/kc_house_data.csv')
# analysis : 


# print(df.info())
# #missing data set = 
# print(df.isnull().sum())

# print(df.describe().transpose())
# print(df.head())

# visualization 
# most houses are in the range of 0 to 1.5 million
# sns.distplot(df['price'])
# plt.show()

# sns.countplot(df['bedrooms'])
# print(df.corr()['price'].sort_values())
# sns.scatterplot(x='price' , y='sqft_living',data=df)
# sns.scatterplot(x='price' , y='bathrooms',data=df)
# sns.barplot(x='bedrooms' , y='price',data=df)
# sns.scatterplot(x='price' , y='long',data=df)
# plt.show()

# sns.scatterplot(x='price' , y='lat',data=df)
# plt.show()

# sns.scatterplot(x='long' , y='lat',data=df,hue='price')
# plt.show()

# print(df.sort_values('price',ascending=True).head(20))

non_top_1_percent = df.sort_values('price',ascending=False).iloc[216:]
# print(non_top_1_percent.head())
# sns.scatterplot(x='long' , y='lat',data=non_top_1_percent,edgecolor=None,alpha=0.2,palette='RdYlGn',hue='price')
# plt.show()

# sns.scatterplot(y='waterfront',x='price',data=non_top_1_percent)
# plt.show()



# feature Engineering 


df = df.drop('id',axis=1)
df['date'] = pd.to_datetime(df['date'])
# print(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

# print(df['year'])
# print(df.head())

# sns.scatterplot(x='month',y='price',data=df)
# plt.show()

# df.groupby('month').mean()['price'].plot()
# df.groupby('year').mean()['price'].plot()

# plt.show()

df = df.drop('date',axis=1)
# print(df.info()) 
df = df.drop('zipcode',axis=1)

X = df.drop('price',axis=1).values
y = df['price'].values

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# scaling 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#fitting
model  = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))
# adam optimizer is used
model.compile(optimizer='adam', loss = 'mse')
#more batch size - more efficiany - more time to train
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size = 128,epochs=400)

model_losses= pd.DataFrame(model.history.history)
model_losses.plot()
plt.show()
predictions = model.predict(X_test)
print("rt mean sq error : ",mean_squared_error(y_test,predictions)**0.5)
print(" mean abs error : ",mean_absolute_error(y_test,predictions))
print("explained_variance_score : ",explained_variance_score(y_test,predictions))

# plt.scatter(y_test,predictions)
# plt.show()

single_house = df.drop('price',axis = 1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
main_pred = model.predict(single_house)
print(main_pred)
