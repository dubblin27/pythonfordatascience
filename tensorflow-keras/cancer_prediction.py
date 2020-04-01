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


df = pd.read_csv('TensorFlow_FILES\DATA\cancer_classification.csv')
print(df.info())