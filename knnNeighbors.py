#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:08:52 2020

@author: alexisng
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import statistics as stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



#load Wisconsin breast cancer data 
cancer = load_breast_cancer()
# print(cancer.DESCR)

#import data set from scikit
X, t = load_breast_cancer(return_X_y=True)
# print(X.shape) #(569, 30)
N = len(X)  # number of rows
# print(N) #(569,)

#train/test split 
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/5, random_state = 5007)
# print(X_train.shape) 
# print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set
# print(N, M) #(455, 144)

#Normalize the feature - SLIDE 22 IN TOPIC 2 
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :]) 

#-----------------------------------------------------#

#sample 
# X_train = np.array([[1,2],[2,3],[3,1],[3,4],[2,3],[4,0]])
# X_test = np.array([[1,4],[4,3],[0,4]])
# t_train = np.array([0,0,0,1,1,1])
# t_test = np.array([1,0,1])
# M = len(X_test) #3
# N = len(X_train) #6
# print((M,N))

#compute distances and sort
def distance(X1,X2):
    global ind
    
    M = len(X2)
    N = len(X1)
    
    dist = np.zeros((M,N))
    ind = np.zeros((M,N)) 
    u = np.arange(N)       # array of numbers from 0 to N-1
    for j in range(M):
        ind[j,:] = u
        
    for j in range(M): #each test point
        for i in range(N): #each training point
            z = X1[i,:]-X2[j,:]
            dist[j,i] = np.dot(z,z)
    ind = np.argsort(dist)
    
    return ind
    
#compute prediction
def prediction(kNN,X_test,t_train):    
    global y 
    # if kNN == 0:
    #     print("k value not valid")
        
    M = len(X_test)
    y = []
    
    for j in range(M):
        temp=[]
        for i in range(kNN):
            temp.append(t_train[ind[j,i]])
        #print(temp)
        mode = stats.mode(temp)
        val = int(mode)
        y.append(val)
    
    return y

def error(t_test):
    global err

    M = len(t_test)
    #print(y) 
    #rint(t_test)
    z = y - t_test
    #print(z)
    err = np.count_nonzero(z)/M
    
    return err
    
def kNN_KFoldFunc(X, kNN):
    
    global averageError
       
    # split data into training and test sets using kfold
    kf = KFold(n_splits = 5, random_state = 5007, shuffle = True)
    
    avgError = []

    for train, test in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_kf, X_test_kf = X[train], X[test]
        t_train_kf, t_test_kf = t_train[train], t_train[test] 
    
        distance(X_train_kf,X_test_kf)
        
        prediction(kNN,X_test_kf,t_train_kf)
        
        error(t_test_kf)
        avgError.append(err)
        print("cross-validation error: " + str(err))
        
    averageError = np.mean(avgError)
    return averageError

def precision_recall_F1(t_test):   
    global precision, recall, f1_score
    global TP, FN, FP, TN

    size = len(t_test)
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    for i in range(size):
        if (t_test[i] == 1 and y[i] == 1): #true positive
            TP += 1 
    
        if (t_test[i] == 1 and y[i] == 0): #truly positive but predicted negative
            FN += 1 
    
        if (t_test[i] == 0 and y[i] == 1): #truly negative but predicted positive
            FP += 1 
    
        if (t_test[i] == 0 and y[i] == 0): #true negative 
            TN += 1 
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*recall*precision/(precision+recall)
    
    return precision, recall

# #------------------------------------------------#    
kNN_errors = []

#implement k-nearest neighbour classifier for each k = 1,2,3,4,5
for k in range(1,6):
    
    #kFold implementation    
    kNN_KFoldFunc(X_train, k)
    print("For k = " + str(k) + ", the average cv error is " + str(averageError)) #
    print("")
    kNN_errors.append(averageError)

#choose best k classifier
index_min = (np.argmin(kNN_errors)) + 1
print("The best k is : " + str(index_min))
    
#Use best classifier to compute test error 
distance(X_train,X_test)
prediction(index_min,X_test,t_train) 
precision_recall_F1(t_test)
error(t_test)

print("The test error is " + str(err)) #print error
print("The F1 score is: " + str(f1_score))

print('')
print('-------------------------------------------------------')    
print('') 
#------------k-NN USING USING SCIKIT-LEARN-----------------------------------------------
from sklearn.metrics import f1_score
#choose best k neighbour 
k_range = range(1,6)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train, t_train, cv = 5)
    # print(1-scores)
    # print('')
    k_scores.append(1-scores.mean())

print("Average cv errors are: " + str(k_scores))
index = np.argmin(k_scores) + 1
print("The best k calculated by scikit-learn is: " + str(index))

#take best classifier and find test error    
knnModel = KNeighborsClassifier(n_neighbors = index)

knnModel.fit(X_train,t_train)
y_pred = knnModel.predict(X_test)
lr_f1 = f1_score(t_test, y_pred)
knnScore = knnModel.score(X_test, t_test)
testError = (1-knnScore)

print("The test error calculated by scikit-learn KNeighborsClassifier is: " + str(testError))
print("The F1 score calculated by SKlearn is: " + str(lr_f1))
#------------------------------------------------------#

# kNN_errors = []

# for k in range(1,6):
#     KF = KFold(n_splits = 5, random_state = 5007, shuffle = True)
    
#     avgerror = []
    
#     for train, test in KF.split(X_train):
    
#         X_train_KF, X_test_KF = X_train[train], X_train[test]
#         t_train_KF, t_test_KF = t_train[train], t_train[test]
        
#         M = len(X_test_KF)
#         N = len(X_train_KF)
#         dist = np.zeros((M,N))
#         ind = np.zeros((M,N)) 
#         u = np.arange(N)       # array of numbers from 0 to N-1
#         #print(u)
#         for j in range(M):
#             ind[j,:] = u
        
#         #compute distances and sort
#         for j in range(M): #each test point
#             for i in range(N): #each training point
#                 #z = X_train[i]-X_valid[j] # just one feature
#                 z = X_train_KF[i,:]-X_test_KF[j,:]
#                 dist[j,i] = np.dot(z,z)
#                 #ind[j,:] = np.argsort(dist[j,:])
#         ind = np.argsort(dist)
#         #print(ind)
        
#         #compute predictions and errors
#         y = []
#         for j in range(M):
#             temp=[]
#             for i in range(k):
#                 temp.append(t_train_KF[ind[j,i]])
#             #print(temp)
#             mode = stats.mode(temp)
#             val = int(mode)
#             y.append(val)
        
#         z = y - t_test_KF
#         # print(z)
#         err = np.count_nonzero(z)/M
#         avgerror.append(err)
#         print("cross-validation error: " + str(err))
    
    
#     meanError = np.mean(avgerror)
#     print("For k = " + str(k) + ", the average cv error is " + str(meanError))
#     kNN_errors.append(meanError)
#     print('')

# index_min = (np.argmin(kNN_errors)) + 1
# print("The best k is : " + str(index_min))

# #Use best classifier to compute test error 

# M = len(X_test) #number rows in test set
# N = len(X_train) #number rows in train set

# dist = np.zeros((M,N))
# ind = np.zeros((M,N)) 
# u = np.arange(N)       # array of numbers from 0 to N-1
# #print(u)
# for j in range(M):
#     ind[j,:] = u

# #compute distances and sort
# for j in range(M): #each test point
#     for i in range(N): #each training point
#         #z = X_train[i]-X_valid[j] # just one feature
#         z = X_train[i,:]-X_test[j,:]
#         dist[j,i] = np.dot(z,z)
#         #ind[j,:] = np.argsort(dist[j,:])
# ind = np.argsort(dist)
# #print(ind)
   

# # compute predictions and error with chosen k
# y = np.zeros(M) # initialize array of predictions

# for j in range(M):
#     y[j] = t_train[ind[j,0]]

# # print(t_test)
# z = y - t_test
# # print(z)
# err = np.count_nonzero(z)/M  #mislassification rate
# print("The test error is " + str(err))



# for k in range(1,6):
#     y = []
#     for j in range(M):
#         temp=[]
#         for i in range(k):
#             temp.append(t_train[ind[j,i]])
#         #print(temp)
#         mode = stats.mode(temp)
#         val = int(mode)
#         y.append(val)
    
#     z = y - t_test
#     # print(z)
#     err = np.count_nonzero(z)/M
#     print("For k = " + str(k) + " is " + str(err))

# print("")

# # compute predictions and error with 1NN (closest 1 point)
# y = np.zeros(M) # initialize array of predictions
# for j in range(M):
#     y[j] = t_train[ind[j,0]]
# # print(y)
# # print(t_test)
# z = y - t_test
# # print(z)
# err = np.count_nonzero(z)/M  #mislassification rate
# print("1NN error is : " + str(err))

