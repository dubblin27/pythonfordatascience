import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

df= pd.read_csv('TensorFlow_FILES/DATA/fake_reg.csv')
print(df.head())

#analysis
# sns.pairplot(df)
# plt.show()

print(df.info())
X = df[['feature1','feature2']].values
y = df['price'].values
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# print(X_train.shape)
# print(X_test.shape)

# to normalize
# if we have large values in our dataset..it could cause errors with the weights 
# so we normalize and scale
# we fit only the X_train as ..we dont want any data leakage from the test set as it is set for test
scaler = MinMaxScaler()
scaler.fit(X_train)
#transform to convert them between the range of 0 to 1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# creation of a neural network 

model = Sequential()
#type of activation function used - RELU = Rectified Linear Unit
model.add(Dense(4,activation='relu')) #1st layer 
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
#3 ways to compiler 

    # Regression Loss Functions
    #     Mean Squared Error Loss
    #     Mean Squared Logarithmic Error Loss
    #     Mean Absolute Error Loss
    # Binary Classification Loss Functions
    #     Binary Cross-Entropy
    #     Hinge Loss
    #     Squared Hinge Loss
    # Multi-Class Classification Loss Functions
    #     Multi-Class Cross-Entropy Loss
    #     Sparse Multiclass Cross-Entropy Loss
    #     Kullback Leibler Divergence Loss


# main 
# 1 . Multi-Class Classification  
#     optimizer = 'rmsprop' , loss = 'catogorical_crossentropy'
# 2 . Binary Classification
#     optimizer = 'rmsprop' , loss = 'binary_crossentropy'
# 3 . Regression Loss Functions 
#     optimizer = 'rmsprop' , loss = 'mse' 
# mse = mean squared error 





model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=250)
# visualizing the loss
# loss_history = pd.DataFrame(model.history.history)
# loss_history.plot()
# plt.show()


mean_sq_error_for_test = model.evaluate(X_test,y_test,verbose = 0)
print(mean_sq_error_for_test)


mean_sq_error_for_train = model.evaluate(X_train,y_train,verbose = 0)
print(mean_sq_error_for_train)

test_predictions =model.predict(X_test)
# print(test_predictions) 
test_predictions = pd.Series(test_predictions.reshape(300,))
pref_df = pd.DataFrame(y_test,columns=['test true Y'])
pref_df = pd.concat([pref_df,test_predictions],axis=1)
pref_df.columns = ['test true Y','model pred']
#true price vs value prediction 
sns.scatterplot(x='test true Y', y='model pred',data= pref_df)
# plt.show()

mae = mean_absolute_error(pref_df['test true Y'],pref_df['model pred'])
mse = mean_squared_error(pref_df['test true Y'],pref_df['model pred']) 
rmse = mean_squared_error(pref_df['test true Y'],pref_df['model pred']) **0.5

print(mae,mse,rmse) 

new_gem = [[998,1000]]
new_gem = scaler.transform(new_gem)
prediction1 = model.predict(new_gem)
print(prediction1)