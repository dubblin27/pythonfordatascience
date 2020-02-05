import numpy as np 
import pandas as pd 
from numpy.random import randn
from numpy.random import randint 
try :
    x
    l  = [1,2,3,4]
    labels = ['a','b','c','d']
    arr = np.array(l,dtype='int')
    d = {
        'a': 10,
        'b': 20,
        'c': 30
    }
    print(pd.Series(l))
    print(pd.Series(l,labels))
    print(pd.Series(d))
    print(pd.Series(sum(l)))

    s1 = pd.Series([1,2,3,4],['a','b','c','d'])
    s2 = pd.Series([7,5,32,54],['a','g','c','d'])

    print(s1)
    print(s2)
    print(s1+s2)

    #data frames

    print(np.random.seed(101))
    df = pd.DataFrame(randint(0,18,[4,4]),['a','b','c','d'],['w','x','y','z'])

    print(df[['w','z']])

    df['addd'] = df['w'] + df['z']
    print(df)
    print(df.drop('addd',axis = 1))
    print(df)
    df.drop('addd', axis = 1,inplace=True)
    print(df)

    print(df.drop('d'))
    print(df)
    df.drop('d',inplace = True)
    print(df)

    #selction of dataframe 
    print(df.loc['a'])
    print(df.iloc[1])

    print(df.loc[['a','b'],['x','y']])
    #op    
    #     w   z
    # a  11  11
    # b  15   8
    # c   4  14
    # d   5  17

    #     w   x   y   z  addd
    # a  11  17   6  11    22
    # b  15   9  13   8    23
    # c   4   8   0  14    18
    # d   5  12   8  17    22

    #     w   x   y   z
    # a  11  17   6  11
    # b  15   9  13   8
    # c   4   8   0  14
    # d   5  12   8  17

    #     w   x   y   z  addd
    # a  11  17   6  11    22
    # b  15   9  13   8    23
    # c   4   8   0  14    18
    # d   5  12   8  17    22

    #     w   x   y   z
    # a  11  17   6  11
    # b  15   9  13   8
    # c   4   8   0  14
    # d   5  12   8  17

    #     w   x   y   z
    # a  11  17   6  11
    # b  15   9  13   8
    # c   4   8   0  14
    #     w   x   y   z

    # a  11  17   6  11
    # b  15   9  13   8
    # c   4   8   0  14
    # d   5  12   8  17

    #     w   x   y   z
    # a  11  17   6  11
    # b  15   9  13   8
    # c   4   8   0  14

    # w    11
    # x    17
    # y     6
    # z    11
    # Name: a, dtype: int64

    # w    15
    # x     9
    # y    13
    # z     8
    # Name: b, dtype: int64

    #     x   y
    # a  17   6
    # b   9  13

    #conditional selection (df)
    df = pd.DataFrame(randint(0,18,[4,4]),['a','b','c','d'],['w','x','y','z'])
    print(df)

    z = df > 6
    print(z)
    print(df[z])

    # #op
    #     w   x   y   z
    # a   5  15  17   2
    # b  16   1   0   5
    # c   5   1  12  10
    # d   5   6   6   4
    #        w      x      y      z
    # a  False   True   True  False
    # b   True  False  False  False
    # c  False  False   True   True
    # d  False  False  False  False
    #       w     x     y     z
    # a   NaN  15.0  17.0   NaN
    # b  16.0   NaN   NaN   NaN
    # c   NaN   NaN  12.0  10.0
    # d   NaN   NaN   NaN   NaN

    print(df[df>8])
    print(df[df['w']>5])

    print(df[df['w']<5])
    #       w     x     y     z
    # a   NaN  13.0  17.0   NaN
    # b  15.0   9.0  16.0  16.0
    # c   9.0   NaN  11.0  17.0
    # d  14.0  11.0   NaN   NaN

    #     w   x   y   z
    # b  15   9  16  16
    # c   9   3  11  17
    # d  14  11   7   8

    #    w   x   y  z
    # a  0  13  17  3
    print("ok")
    print(df[df['w']>5][['y','z']])
    #     y   z
    # a   3   6
    # c  14  15
    # d   6   8



except:
    a =10 

#conditional selection (df)
df = pd.DataFrame(randint(0,18,[4,4]),['a','b','c','d'],['w','x','y','z'])
print(df)
print("\n")

print(df['w'],(df['w']>5),sep = "\n")

print("\n")

print(df['z'],(df['z']>7),sep = "\n")
print("final")
print((df['w']>5) & (df['z']>7))
print(df[(df['w']<5) & (df['z']<7)])
print(df[(df['w']>5) & (df['z']>7)])

# #op:
#     w   x   y   z
# a  13  12   2  14
# b  15   4   1  14
# c   1  17   2   5
# d  15  14  10  13


# a    13
# b    15
# c     1
# d    15
# Name: w, dtype: int64
# a     True
# b     True
# c    False
# d     True
# Name: w, dtype: bool


# a    14
# b    14
# c     5
# d    13
# Name: z, dtype: int64
# a     True
# b     True
# c    False
# d     True
# Name: z, dtype: bool
# final
# a     True
# b     True
# c    False
# d     True
# dtype: bool
#    w   x  y  z
# c  1  17  2  5
#     w   x   y   z
# a  13  12   2  14
# b  15   4   1  14
# d  15  14  10  13