#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:41:36 2020

@author: alexisng
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import statistics as stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#--------------------FUNCTIONS------------------------#
def sigmoid(z):
    global y
    y = 1 / (1+np.exp(-z))
    return y

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

#-----------------------------------------------------#   
 
#load Wisconsin breast cancer data 
cancer = load_breast_cancer()
# print(cancer.DESCR)

#import data set from scikit
X, t = load_breast_cancer(return_X_y=True)
N = len(X)  

#train/test split 
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/5, random_state = 5007)
test_size = len(X_test) #114
train_size = len(X_train) #455


#Normalize the feature - SLIDE 22 IN TOPIC 2 
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :]) 


#GRADIENT DESCENT
new_col = np.ones(train_size)
X1_train = np.insert(X_train,0,new_col,axis = 1)
alpha = 1 #learning rate

#initial parameter values
w = np.zeros(X1_train.shape[1])

z = np.zeros(N)
IT = 500 #iteration t 
gr_norms = np.zeros(IT) #to store squared norm of gradient at each iteration 
cost = np.zeros(IT) #to store the cost at each 

for n in range(IT):
    z = np.dot(X1_train,w) 
    sigmoid(z) #sigmoid function
    diff = y - t_train
    gr = np.dot(X1_train.T, np.transpose(diff.T))/N #computation of the gradient
    
    #compute squared norm of the gradient
    gr_norm_sq = np.dot(gr, gr)
    gr_norms[n] = gr_norm_sq
    
    #update the vector of parameters
    w = w-alpha*gr
    
    #compute the cost (average loss)
    cost[n] = 0
    for i in range(train_size):
        cost[n] += t_train[i]*np.logaddexp(0, -z[i]) + (1-t_train[i])*np.logaddexp(0,z[i])

    cost[n] = cost[n]/N
    
print("w: " + str(w))
# print("cost of first 5 vectors: " + str(cost[:5]))
# print("cost of last 5 vectors: " + str(cost[IT-5:IT]))
# print("gradient normal: " + str(gr_norms[IT-5:IT]))


# # plot change of cost during gradient descent
# plt.figure()
# lin = np.linspace(1,IT,IT)
# plt.plot(lin, cost, color = 'blue')
# plt.title('Change in Cost function')
# plt.xlabel('cost')
# plt.ylabel('iteration number')
# #plt.scatter(lin, gr_norms, color = 'red')
# plt.show()

#COMPUTE TEST ERROR
new_col = np.ones(test_size)
X1_test = np.insert(X_test, 0, new_col, axis = 1) 
z = np.dot(X1_test,w)
y = np.zeros(test_size)

#the classifier that minimizes the misclassification rate
for i in range(test_size):
    if(z[i]>=0):
        y[i]=1
u = y - t_test
err = np.count_nonzero(u)/test_size #misclassification rate
print("The misclassification rate for when threshold = 0 is: " + str(err))

#compute precision and recall (0 indicates benign, 1 indicates malignant
precision_recall_F1(t_test)

table = np.array([[TN,FP],[FN,TP]])                

print(table)
print("The F1 score is: " + str(f1_score))

#other classifiers obtained with other thresholds 
precision_array = []
recall_array = []

#sorting z
z_temp = np.argsort(z)
z1 = np.zeros(len(z))
for i in range(len(z)):
    z1[i] = z[z_temp[i]]

# print(z1[:10])

for j in z1:
    y = np.zeros(test_size)
    for i in range(test_size):
        if(z[i]>=j):
            y[i]=1
    u = y - t_test
    err = np.count_nonzero(u)/test_size #misclassification rate

    #print("The misclassification rate for threshold " + str(j) + " is: " + str(err))
    
    precision_recall_F1(t_test)
    
    #update arrays for graphing 
    precision_array.append(precision)
    recall_array.append(recall)    
    
    # print(precision, recall, f1_score)

plt.figure()
plt.plot(recall_array, precision_array)
plt.title('Precision-Recall Curve')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

# # print(z[:10])
# # print(y[:10])
# # print(t_test[:10])
# # print(u[:10])

print('')
print('-------------------------------------------------------')    
print('') 

#------------Logistic Regression USING USING SCIKIT-LEARN-----------------------------------------------

logisticModel = LogisticRegression(random_state = 5007)
logisticModel.fit(X_train, t_train)

y_pred = logisticModel.predict(X_test)

w = logisticModel.coef_
print("sklearn w: " + str(w))
print('')
logisticScore = logisticModel.score(X_test,t_test)
testError = (1-logisticScore)
print("The test error calculated by SKlearn Logistic Regression is: " + str(testError))
print('')

logisticMatrix = confusion_matrix(t_test,y_pred)
confusion_df = pd.DataFrame(logisticMatrix)

confusion_df = pd.DataFrame(confusion_matrix(y_pred, t_test),
             columns=["Predicted Class " + str(cancer.target_names) for cancer.target_names in [0,1]],
             index = ["Class " + str(cancer.target_names) for cancer.target_names in [0,1]])

print(confusion_df)
print('')
# print(classification_report(t_test,y_pred))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

#predict probabilities
logistic_probs = logisticModel.predict_proba(X_test)
logistic_probs = logistic_probs[:,1] #positives only 

lr_precision, lr_recall, _ = precision_recall_curve(t_test, logistic_probs)
lr_f1 = f1_score(t_test, y_pred)
print("The F1 score calculated by SKlearn is: " + str(lr_f1))

plt.figure()
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
plt.title('Precision-Recall Curve using SKlearn')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

# f1_score = f1_score(t_test,y_pred)
# print(f1_score)
# recall = recall_score(t_test,y_pred)
# print(recall)
# precision = precision_score(t_test,y_pred)
# print(precision)
