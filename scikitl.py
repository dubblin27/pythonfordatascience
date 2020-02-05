import numpy as np 
from sklearn.model_selection import train_test_split 
import sklearn.linear_model 
x,y = np.arange(10).reshape((5,2)),range(5)

x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size = 0.3)
# print(x_train) 
# print(".")
# print(x_test)
# print(".")
# print(y_train)
# print(".")
# print(y_test)
model = linear_model()
model.fit(x_train,y_train)