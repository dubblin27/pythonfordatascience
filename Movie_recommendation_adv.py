#colloborative filtering will be used 
# model based CF using singular value Decomposition and 
# Memeory based Decomposition by cosine simularity 
from math import sqrt
import numpy as np 
import pandas as  pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp 
from scipy.sparse.linalg import svds

columns_name = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('dataset and ipynb/19-Recommender-Systems/u.data',sep='\t',names=columns_name)
# print(df.head())

# we only have the ID of the movies 
movie_titles = pd.read_csv('dataset and ipynb/19-Recommender-Systems/Movie_Id_Titles')

df = pd.merge(df,movie_titles,on='item_id')
# print(df.head())

#to find unique users and movies 

n_users = df.user_id.nunique()

n_items = df.item_id.nunique()
print(n_items,"\n",n_users)

#split training and testing of data
train_data,test_data = train_test_split(df,test_size = 0.25 )

#memory based colloborative filtering
# Memory-Based Collaborative Filtering approaches can be divided into two main sections: 
# user-item filtering and item-item filtering.

# A user-item filtering will take a particular user, 
# find users that are similar to that user based on similarity of ratings, and 
# recommend items that those similar users liked.

# In contrast, item-item filtering will take an item, 
# find users who liked that item, and find other items that those users or 
# similar users also liked. It takes items and outputs other items as recommendations.

# Item-Item Collaborative Filtering: “Users who liked this item also liked …”
# User-Item Collaborative Filtering: “Users who are similar to you also liked …”

#eg : netflix 



# In both cases, you create a user-item matrix which built from the entire dataset.

# Since we have split the data into testing and training we will need to create two [943 x 1682] matrices (all users by all movies).

# The training matrix contains 75% of the ratings and the testing matrix contains 25% of the ratings.



# now to create 2 matrices - train and test 
train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3] 

# to calculate Cosine Similarity 
#op will range from 0-1 as the ratings are all positive 
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(test_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user' :
        mean_user_rating = ratings.mean(axis = 1)
        #np.newaxis - mean_user_ratings have same format as the rating 
        ratings_diff = (ratings - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item' :
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#evaluation - using Root mean squared error 
# prediction[ground_truth.nonzero()] - only to obtain the ratings that are in the test data set
def rmse(prediction,ground_truth) :
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth)) 

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# draw backs of memory based algo are not scalable to the real world scnerio 

# model colloborative filtering 

sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

# factorization method using single value decomposition 
# U is an (m x r) orthogonal matrix
# S is an (r x r) diagonal matrix with non-negative real numbers on the diagonal
# V^T is an (r x n) orthogonal matrix

# Elements on the diagnoal inSare known as *singular values of X
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))

