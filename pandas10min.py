import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from numpy.random import randn
from numpy.random import randint 
from random_word import RandomWords 
r = RandomWords()
#One-dimensional ndarray with axis labels
s = pd.Series([1,3,5,np.nan,6,8]) 
print(s)
# op
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64

print( *(pd.date_range('20130101',periods=6)),sep = "\n")
# op
# 2013-01-01 00:00:00
# 2013-01-02 00:00:00
# 2013-01-03 00:00:00
# 2013-01-04 00:00:00
# 2013-01-05 00:00:00
# 2013-01-06 00:00:00 

df = pd.DataFrame(np.random.randint(10,20,[6,4]),index = list('pqrstv'), columns=list('abcd') )
#     A   B   C   D
# P  13  11  11  16
# Q  13  19  17  13
# R  16  19  19  12
# S  17  13  18  17
# t  17  10  18  16  
# v  11  18  19  11
# df = pd.DataFrame(randint(0,18,[4,4]),['a','b','c','d'],['w','x','y','z'])

print(df)
c = r.get_random_words()
d = c[0:4]

df2 = pd.DataFrame({
    'a' : 1. ,
    'b' : pd.Timestamp('20130102') ,
    'c' : pd.Series(np.random.randn(4)),
    'd' : pd.Categorical(d)

    

})

print(df2)
print(df2.dtypes)

# op
#      a          b         c            d
# 0  1.0 2013-01-02  0.322951       biting
# 1  1.0 2013-01-02 -0.245057    associate
# 2  1.0 2013-01-02  1.173008   fierceness
# 3  1.0 2013-01-02  0.431466  confederate
# a           float64
# b    datetime64[ns]
# c           float64
# d          category
# dtype: object

print(df.head()) #insert any value inside 
print(df.tail()) #insert any value inside

# op

#     A   B   C   D
# P  10  18  19  17
# Q  15  13  15  12
# R  15  17  13  12
# S  16  15  13  13
# t  11  12  10  19
#     A   B   C   D
# Q  15  13  15  12
# R  15  17  13  12
# S  16  15  13  13
# t  11  12  10  19
# v  17  11  13  17

print(df.index)
print(df.columns)
print(df.values) 
print(df.describe())
print(df.T)

# op
# Index(['P', 'Q', 'R', 'S', 't', 'v'], dtype='object')
# Index(['A', 'B', 'C', 'D'], dtype='object')
# [[11 12 11 15]
#  [15 14 17 13]
#  [14 16 15 13]
#  [19 17 10 19]
#  [11 17 19 19]
#  [19 11 11 15]]
#                A          B          C          D
# count   6.000000   6.000000   6.000000   6.000000
# mean   14.833333  14.500000  13.833333  15.666667
# std     3.600926   2.588436   3.710346   2.732520
# min    11.000000  11.000000  10.000000  13.000000
# 25%    11.750000  12.500000  11.000000  13.500000
# 50%    14.500000  15.000000  13.000000  15.000000
# 75%    18.000000  16.750000  16.500000  18.000000
# max    19.000000  17.000000  19.000000  19.000000
#     P   Q   R   S   t   v
# A  11  15  14  19  11  19
# B  12  14  16  17  17  11
# C  11  17  15  10  19  11
# D  15  13  13  19  19  15
print(df)
print(df.sort_index(axis=0,ascending=False,kind='mergesort'))
print(df.sort_index(axis=1,ascending=False,kind='mergesort'))

print(df[['a','b']])
df['sum'] = df['a'] +df['b'] +df['c'] +df['d'] 
print(df)
# df.drop(df.columns[[1,2]], axis= 1,inplace=True)
df.drop(df.iloc[:,1:3], axis= 1,inplace=True) #to drop specific columns
print(df)

# op 
#     a   d  sum
# p  16  13   53
# q  16  13   58
# r  18  14   57
# s  11  15   59
# t  19  13   60
# v  12  17   58

df2.loc[b[0]]