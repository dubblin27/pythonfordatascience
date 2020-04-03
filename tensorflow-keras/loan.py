# TensorFlow_FILES\DATA\lending_club_info.csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('TensorFlow_FILES\DATA\lending_club_loan_two.csv') 
# print(df1.head())
# print(df1.info())

sns.countplot(x = 'loan_status', data = df)
plt.show()