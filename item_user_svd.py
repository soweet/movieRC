# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:06:16 2017

@author: sj
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
#from scipy.spatial.distance import cosine
from numpy import linalg as la

def cos_corrMat(data_matrix, flag='user'):  #calculate cosin similarity
    if flag == 'user':
        n_num = data_matrix.shape[0]
    elif flag == 'item':
        n_num = data_matrix.shape[1]
        data_matrix = data_matrix.T
    similarity_mat = np.zeros([n_num,n_num])
    for i in range(n_num):
        for j in range(i,n_num):
            #if(i==j):
               # similarity_mat[i,j] = 1.0
            if(i!=j):
                overlap = np.nonzero(np.logical_and(data_matrix[i,:]>0,data_matrix[j,:]>0))[0]
                if len(overlap)==0:
                    similarity_mat[i,j] = 0.0
                else:
                    l=data_matrix[i,overlap]
                    r=data_matrix[j,overlap]
                    cos = float(l.dot(r.T)/(la.norm(l)*la.norm(r)))
#                    similarity_mat[i,j] = cos
                    similarity_mat[i,j] = 0.5+0.5*cos
                    #similarity_mat[i,j] = 0.5+0.5*cosine(l,r)
                    #similarity_mat[i,j] = cosine(l,r)
    similarity_mat = similarity_mat+similarity_mat.T+np.eye(n_num)
    return similarity_mat

def preMat(data_matrix,similarity_mat, flag='user'): #prediect rating scores
    if flag == 'user':
        ratsim_sum = similarity_mat.dot(data_matrix)
        sim_sum = np.zeros_like(data_matrix)
        for j in range(data_matrix.shape[1]):
            overlap = np.nonzero(data_matrix[:,j]>0)[0]
            for i in range(data_matrix.shape[0]):
                sim_sum[i,j] = np.sum(similarity_mat[i,overlap])
    elif flag == 'item':
        ratsim_sum = data_matrix.dot(similarity_mat)
        sim_sum = np.zeros_like(data_matrix)
        for i in range(data_matrix.shape[0]):
            overlap = np.nonzero(data_matrix[i,:]>0)[0]
            for j in range(data_matrix.shape[1]):
                sim_sum[i,j] = np.sum(similarity_mat[j,overlap])
    sim_sum_zeros = np.nonzero(sim_sum==0)
    pre_mat = ratsim_sum/sim_sum
    pre_mat[sim_sum_zeros]=0
    return pre_mat

def find_SVD_mnum(Sigma):  #fing 90% main element
    Sig2 = Sigma**2
    stand_values = sum(Sig2)*0.9
    for i in range(int(len(Sigma)/4),len(Sigma)):
        if(sum(Sig2[:i])>stand_values):
            break
    return i


def SVD_pre(data_matrix):  #SVD prediect rating scores
    U,Sigma,VT = la.svd(data_matrix)
    svd_mnum = find_SVD_mnum(Sigma)
    Sigma_main = np.mat(np.eye(svd_mnum)*Sigma[:svd_mnum])  #build a duijiao matrix
    formedItems = ((data_matrix.T).dot(U[:,:svd_mnum])).dot(Sigma_main.I)  #shape:(1682,m)
    svd_item_similarity = pairwise_distances(formedItems, metric='cosine')
    SVD_pre = preMat(data_matrix,svd_item_similarity, flag='item')
    return SVD_pre
    
    
def rmse(prediction, ground_truth): #calculate the rmse
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
    
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('movielens/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = {0} | Number of movies = {1}'.format(n_users,n_items))

train_data, test_data = cv.train_test_split(df, test_size=0.25)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  
 
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#CF
user_similarity = cos_corrMat(train_data_matrix, flag='user')
item_similarity = cos_corrMat(train_data_matrix, flag='item')

item_prediction = preMat(train_data_matrix, item_similarity, flag='item')
user_prediction = preMat(train_data_matrix, user_similarity, flag='user')

#SVD
svd_pre = SVD_pre(train_data_matrix)

print('User-based CF RMSE:{0}'.format(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE:{0}'.format(rmse(item_prediction, test_data_matrix)))
print('SVD RMSE:{0}'.format(rmse(svd_pre, test_data_matrix)))
