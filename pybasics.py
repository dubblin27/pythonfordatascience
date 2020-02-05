# a=10
# b="sabrish"
# print("i am {} and my name is {}".format(a,b))
# print("i am {x} and my name is {y}".format(x=a,y=b))

#dictionaries 

# d = {
#     'key1':[1,2,3],
#     'key2':2
# }
# print(d['key1'][::-1],d['key2'])

# d = {
#     'k1' : {
#         'k2':[1,2,3]

#     }
# }
# print(d['k1'])
# print(d['k1']['k2'])
# print(d['k1']['k2'][0])


# dict  = {
#     'k1':"hello",
#     'k2':"bye"
# }
# print(dict)
# print(dict.keys())
# print(dict.items())
# print(dict.values())
# # op 
# {'k1': 'hello', 'k2': 'bye'}
# dict_keys(['k1', 'k2'])
# dict_items([('k1', 'hello'), ('k2', 'bye')])
# dict_values(['hello', 'bye'])

#tuples - immutable
# t = (1,2,3,4)
# print(t)
# print(t[0])
# t[0] = 20
# print(t)
#sets
# c = {1,1,1,1,2,2,2,3,3,3}
# print(c)
# op - {1,2,3}

# c = {1,1,1,1,2,2,2,3,3,3}
# c.add(5)
# c.add(5)
# print(c)


# c = list(range(10))
# print(c)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# c = [i**2 for i in range(1,10)]
# print(c)
# def func(var):return var*10
# c = lambda x:x*20
# z= list(range(1,11))
# print(z)
# d= list(map(func,z))
# f = list(map(c,z))
# print(*d,*f, sep = " \n")


# lambda function 

# normal function :

# def function(x):
#     return x*2 
# # can also be coded as 
# def function(x):return x*2

# lambda function :

# z= lambda x:x*2 #z is the function

# eg:  

# arr = [x for x in range(10)]

# z = list(map(lambda c:c**2,arr))
# print(z)

# # filter function 

# def mul(arr):
#     z= arr 
#     x = []
#     for i in range(len(z)):
#         if z[i]%2 == 0 :
#             x.append(z[i])    
#     return(x)

# arr = [x for x in range(10)]
# print(mul(arr))

# # the above code is replaced by the lower code 

# g = list(filter(lambda x : x%2 == 0 ,arr ))
# print(g)
