
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import statistics as stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('Data/spambase.data')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values 

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/3, random_state = 5007)
# print(X_train.shape) 
# print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set
# print(N, M) 
num = np.arange(100,1001,100)  


#--------------------One Decision Tree Classifier--------------------------#

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

depth = []
#finding max leaf nodes in the range of 2 to 401 
for i in range(2,401):
    model = DecisionTreeClassifier(max_leaf_nodes = i, random_state = 5007) 
    model = model.fit(X_train,t_train)
    scores = cross_val_score(model, X_train, t_train, cv = 5) #calculate cv scores
    scores = 1-scores #to obtain the error
    depth.append(scores.mean())

plt.figure()
plt.plot(depth)
plt.title('CV Error Obtained from Decision Tree Model')
plt.xlabel('number of predictors')
plt.ylabel('CV Errors')

no_of_leaves = np.argmin(depth)+2
print(no_of_leaves)    

dec_tree = DecisionTreeClassifier(max_leaf_nodes = no_of_leaves, random_state = 5007)
dec_tree = dec_tree.fit(X_train,t_train)
y_pred = dec_tree.predict(X_test)
score = dec_tree.score(X_test,t_test)
error = 1-score
print("Decision Tree: Test Error: " + str(error))

dec_tree_err = []
baggingErrors = []
randomForestError = []
adaboost1 = []
base_stumps = DecisionTreeClassifier(max_depth = 1, random_state = 5007)
adaboost2 = []
base_dectrees = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 5007)
adaboost3 = []
base_norestr = DecisionTreeClassifier(max_depth = None, max_leaf_nodes = None, random_state = 5007)

for i in range(100,1001,100):
    dec_tree_err.append(error)

#--------------------Bagging Classifier-----------------------------------#

    bagging = BaggingClassifier(n_estimators = i, random_state = 5007)
    bagging = bagging.fit(X_train,t_train)
    y_pred = bagging.predict(X_test)
    bagging_err = bagging.score(X_test,t_test)
    baggingErrors.append(1-bagging_err)

#-------------------Random Forest Classifier------------------------------#

    rand_forest = RandomForestClassifier(n_estimators = i, random_state = 5007)
    rand_forest = rand_forest.fit(X_train,t_train)
    y_pred = rand_forest.predict(X_test)
    rand_forest_err = rand_forest.score(X_test,t_test)
    randomForestError.append(1-rand_forest_err)

#-----------------Adaboost Classifier w/ decision stumps-----------------#

    ada_stumps = AdaBoostClassifier(base_estimator = base_stumps, n_estimators = i, random_state = 5007)
    ada_stumps.fit(X_train,t_train)
    y_pred = ada_stumps.predict(X_test)
    ada_stumps_err = ada_stumps.score(X_test,t_test)
    adaboost1.append(1-ada_stumps_err)
    #print("Adaboost Classifier with Decision Stumps: Test Error = " + str(1-ada_stumps_err))

#-----------------Adaboost Classifier w/ decision trees (10 leaves)------#

    ada_restrict = AdaBoostClassifier(base_estimator = base_dectrees, n_estimators = i, random_state = 5007)
    ada_restrict.fit(X_train,t_train)
    y_pred = ada_restrict.predict(X_test)
    ada_restrict_err = ada_restrict.score(X_test,t_test)
    adaboost2.append(1-ada_restrict_err)
    

#-----------------Adaboost Classifier w/ decision trees (no restriction)-#

    ada_norestrict = AdaBoostClassifier(base_estimator = base_norestr, n_estimators = i, random_state = 5007)
    ada_norestrict.fit(X_train,t_train)
    y_pred = ada_norestrict.predict(X_test)
    ada_norestrict_err = ada_norestrict.score(X_test,t_test)
    adaboost3.append(1-ada_norestrict_err)


#Plotting#
plt.figure()
plt.plot(num, dec_tree_err, marker = "o", color = 'c', label = "Decision Tree")
plt.plot(num,baggingErrors, marker = "o", color = 'b', label = "Bagging")
plt.plot(num,randomForestError, marker = "o", color = 'r', label = "Random Forest")
plt.plot(num,adaboost1, marker = "o", color = 'g', label = "Adaboost with decision stumps")
plt.plot(num,adaboost2, marker = "o", color = 'k', label = "Adaboost with decision trees (max_leaf_nodes = 10")
plt.plot(num,adaboost3, marker = "o", color = 'y', label = "Adaboost with decision trees (no restriction)" )
plt.title('Test Errors of 5 Ensemble Methods and Decision Tree')
plt.xlabel('number of predictors')
plt.ylabel('test error')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
