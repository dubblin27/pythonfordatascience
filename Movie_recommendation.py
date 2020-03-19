#portfolio project 
#collobrative Filerting 
#based on item similarity 
#movie Lens Dataset

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

columns_name = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('dataset and ipynb/19-Recommender-Systems/u.data',sep='\t',names=columns_name)

# print(df.head())
movie_titles = pd.read_csv('dataset and ipynb/19-Recommender-Systems/Movie_Id_Titles')
# print(movie_titles.head())
#id + title
df = pd.merge(df,movie_titles,on='item_id')
print(df)

print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())


ratings['num of ratings'].hist(bins=70) 

# plt.show()

ratings['rating'].hist(bins=70)
# plt.show()

sns.jointplot(x='rating',y='num of ratings', data=ratings,alpha=0.5)
# plt.show()


moviemat = df.pivot_table(index='user_id',columns='title', values='rating')
# print(moviemat.head())

print(ratings.sort_values('num of ratings',ascending=False).head(10))
movi = input("enter Movie name : ")
movie_user_ratings1 = moviemat[movi]


# print(movie_user_ratings1.head())

#corewidth..to find the relationship - to find rowise and pairwise correlation between 2 sets 
similar_to_movie1=moviemat.corrwith(movie_user_ratings1)

#to clean NaN 
corr_movie1 = pd.DataFrame(similar_to_movie1,columns=['Correlation'])

corr_movie1.dropna(inplace=True)
#correlation of movie rating with movie1 movie ratings
# print(corr_movie1.head())

#we can get the most similar movies to movie1 

print(corr_movie1.sort_values('Correlation',ascending=False).head(10))

#filtering the movies < 100 rating 

corr_movie1 = corr_movie1.join(ratings['num of ratings'])
# print(corr_movie1.head()) 

print(corr_movie1[corr_movie1['num of ratings']>100].sort_values('Correlation',ascending=False))
