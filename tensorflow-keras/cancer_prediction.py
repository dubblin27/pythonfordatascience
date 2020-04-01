#tensorflow classification 

# early stopping :
# keras can automatically stop training based on a loss condition on validation
#data passes during the model.fit

# dropout layers:
# layers to turn off neurons during the training to avoid overfitting

# each dropout layer will drop a user-defined percentage of neuron unit in the previous layer every batch





import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('TensorFlow_FILES\DATA\cancer_classification.csv')
X = df.drop('benign_0__mal_1',axis=1).values 
y = df['benign_0__mal_1'].values

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)
scaler = MinMaxScaler()
# to prevent data leakage -  we use minmaxscaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
# to detect only a binary classification at last - cancer cells malignent or benign ie 0 or 1 - sigmoid activation function is used
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer='adam')
model.fit(x =X_train, y= y_train, epochs = 600, validation_data = (X_test,y_test))
# visualizatin 
# this is over fitting 
# validation loss is increasing and training loss is decreasing ..this shows we are over fittinig to training dataset
#
# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

# we can use early stopping to stop from overfitting as it trains again 
#mode = min as to minimize the loss
# patience - analize for 25 epochs check for losses ..if needed ..stop training
early_stop = EarlyStopping(monitor = 'val_loss',mode='min',verbose=1,patience=25)
model1 = Sequential()
model1.add(Dense(30,activation='relu'))
model1.add(Dense(30,activation='relu'))
model1.add(Dense(30,activation='relu'))
model1.add(Dense(30,activation='relu'))
# to detect only a binary classification at last - cancer cells malignent or benign ie 0 or 1 - sigmoid activation function is used
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss = 'binary_crossentropy',optimizer='adam')
model1.fit(x =X_train, y= y_train, epochs = 600, validation_data = (X_test,y_test),callbacks=[early_stop])
#early stopping - 46
# model_losses = pd.DataFrame(model1.history.history)
# model_losses.plot()
# plt.show()

#dropout - to turn of a % of neurons 
#rate - prob of randomly turn off the actual neurons -> 0 - 0%, 1 - 100%,  random subselction of neurons
model2 = Sequential()
model2.add(Dense(30,activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(15,activation='relu'))

model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss = 'binary_crossentropy',optimizer='adam')
early_stop1 = EarlyStopping(monitor = 'val_loss',mode='min',verbose=1,patience=25)

model2.fit(x =X_train, y= y_train, epochs = 600, validation_data = (X_test,y_test),callbacks=[early_stop1])
# main_losses = pd.DataFrame(model2.history.history)
# main_losses.plot()
# plt.show()

# new prediction system (better prediction).predic_classes instead of .predict
predictions = model2.predict_classes(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
