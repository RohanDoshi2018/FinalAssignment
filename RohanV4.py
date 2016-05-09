
# coding: utf-8

# In[1]:

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import sqlite3
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn import decomposition
from sklearn.cluster import KMeans
import random

# get_ipython().magic(u'matplotlib inline')

# read in data (x) and labels (y)
data = pd.read_csv("x_named_z.csv", low_memory=False, index_col=0)
labels = pd.read_csv('y_named.csv', header=None) 

x = data
x = x.drop('UNITID', axis=1)

y = pd.read_csv('y_named.csv', low_memory=False, header=None)
schools = y.ix[:,0]
y.index = schools
list_schools = schools.tolist()
y.index = list_schools
y = y.drop(0,axis=1)

# split the data and labels in training and testing segments
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)


# In[2]:

# 3-fold cross validated hyperparameter fine tuning with gridsearchcv

# # Hyperparameter optimization for LinearSVC using Grid Search Cross-Validation
# C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
# max_iter_range = [100, 500, 1000]
# parameters = {"C":C_range, "max_iter":max_iter_range}
# clf = GridSearchCV(lsvc, parameters)
# clf.fit(data_final, labels)
# print("The best classifier is: ", clf.best_estimator_)

# # Hyperparameter optimization for SVC using Grid Search Cross-Validation
# C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
# parameters = {"C":C_range}
# clf = GridSearchCV(svc, parameters)
# clf.fit(data_final, labels)
# print("The best classifier is: ", clf.best_estimator_)

# # Hyperparameter optimization for KNN using Grid Search Cross-Validation
# parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'leaf_size': [10,20,30,40,50,60,70,80,90,100]}]
# clf = GridSearchCV(knn, parameters)
# clf.fit(data_final, labels)
# print("The best classifier is: ", clf.best_estimator_)

# # Hyperparameter optimization for Decision Tree using Grid Search Cross-Validation
# parameters = [{'max_features': ['auto', 'log2'], 'max_depth': [10,20,30,40,50,60,70,80,90,100]}]
# clf = GridSearchCV(dt, parameters)
# clf.fit(data_final, labels)
# print("The best classifier is: ", clf.best_estimator_)

# # Hyperparameter optimization for Random Forest using Grid Search Cross-Validation
# parameters = [{"n_estimators": [5, 10, 20, 50]}]
# clf = GridSearchCV(rf, parameters)
# clf.fit(data_final, labels)
# print("The best classifier is: ", clf.best_estimator_)


# In[3]:

def plotPred(y_predict, y_test, name):
    plt.figure()
    plt.scatter(y_test, y_predict)
    t = np.arange(0, 100000, 1)
    plt.plot(t,t)
    plt.xlabel('Actual incomes')
    plt.ylabel('Predicted incomes')
    plt.title(name)
    plt.show()
    print("RMSE:", rmse(y_test, y_predict))    
    print("r2:", r2_score(y_test, y_predict))

def rmse(y_test, y_predict):    
    a = y_test - y_predict
    a = a ** 2
    a = a.mean()
    a = np.sqrt(a)
    return a 

def plotResiduals(y_predict, y_test, name):
    plt.figure()
    plt.hist(y_test-y_predict)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(name + ' Residuals')
    plt.show()

def listTenLargestResiduals(y_predict, y_test):
    residuals = y_test-y_predict
    residuals_10worst = residuals.sort(1,axis=0)
    print(residuals_10worst.head(n=10))
    residuals_10best = residuals.sort(1,axis=0,ascending=False)
    print(residuals_10best.head(n=10))

# fit and test the regressor, output graph and evaluation statistics
def regress(model, name):
    reg = model
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    y_predict = np.reshape(y_predict, (len(y_predict),1))    
    plotPred(y_predict, y_test, name)
    plotResiduals(y_predict, y_test, name)
    listTenLargestResiduals(y_predict, y_test)
        
regress(linear_model.LinearRegression(), 'Ordinary Least Squares Regression')
regress(KNeighborsRegressor(), "KNN Regressor")
regress(linear_model.Ridge(), 'Ridge Regression')
regress(linear_model.Lasso(), 'Lasso Regression')
regress(linear_model.ElasticNet(), 'Elastic Net Regression')
regress(DecisionTreeRegressor(max_depth=10), 'Decision Tree Regression')
regress(RandomForestRegressor(n_estimators = 100, max_depth = 5, warm_start = False), 'Random Forest Regression')


# In[4]:

# feature importance 

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
coef = pd.DataFrame(reg.coef_.T, index=x.columns.values)
coef = coef.sort(0,axis=0, ascending=False)
coef


# In[5]:

from sklearn.feature_selection import SelectFromModel
print(x.shape)
reg = linear_model.LinearRegression().fit(x, y)
x_new = SelectFromModel(reg, prefit=True).transform(x)
print(x_new.shape)


# In[6]:

# plotting covariance matrix after feature selection
# http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html
x_new = SelectKBest(f_regression, k=21).fit_transform(x, y)
x_norm = []
x_norm[:] = x_new[:]
x_norm = np.asarray(x_norm)
x_norm -= x_norm.mean(axis=0)
x_norm /= x_norm.std(axis=0)

emp_cov = np.dot(x_norm.T, x_norm) / len(x_norm)
vmax = emp_cov.max()
vmin = emp_cov.min()
plt.imshow(emp_cov, interpolation='nearest', vmin=vmin, vmax=vmax,
           cmap=plt.cm.RdBu_r)
plt.xticks(())
plt.yticks(())
plt.title('Empirical Covariance')
plt.show()


# In[10]:

# PCA, number of features vs feature importance
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py
from sklearn import decomposition

pca = decomposition.PCA()
pca.fit(x)

var_explained_cuml = []
var_explained_cuml.append(pca.explained_variance_ratio_[0])

for i in range(1, 30):
    var_explained_cuml.append(var_explained_cuml[i-1]+pca.explained_variance_ratio_[i])
plt.plot(var_explained_cuml, linewidth=2)
plt.axis('tight')
plt.xlabel('Top "n" PCA Components Included')
plt.ylabel('Proportion of Total Explained Variance')


# In[8]:

pca = decomposition.PCA(n_components=2)
pca.fit(x)
x_pca_reduced = pca.fit_transform(x)
x_pca_reduced_pd = pd.DataFrame(data=x_pca_reduced, index=list_schools, columns=['e1', 'e2'])  # 1st row as the column names

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# Harvard University
ax.annotate(x_pca_reduced_pd.index.values[516], xy=(x_pca_reduced_pd.ix[516,'e1'],x_pca_reduced_pd.ix[516,'e2']), 
            xytext=(-2,5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# University of Phoenix - Phoenix Campus
ax.annotate(x_pca_reduced_pd.index.values[31], xy=(x_pca_reduced_pd.ix[31,'e1'], x_pca_reduced_pd.ix[31,'e2']), 
            xytext=(-6,11), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Princeton University
ax.annotate(x_pca_reduced_pd.index.values[726], xy=(x_pca_reduced_pd.ix[726,'e1'], x_pca_reduced_pd.ix[726,'e2']), 
            xytext=(-9.5,2), color='purple', arrowprops=dict(facecolor='cyan', shrink=0.05))

# Massachusetts Maritime Academy
ax.annotate(x_pca_reduced_pd.index.values[525], xy=(x_pca_reduced_pd.ix[525,'e1'], x_pca_reduced_pd.ix[525,'e2']), 
            xytext=(-2.2,-2.6), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# California Institute of the Arts
ax.annotate(x_pca_reduced_pd.index.values[82], xy=(x_pca_reduced_pd.ix[82,'e1'], x_pca_reduced_pd.ix[82,'e2']), 
            xytext=(-1.3,8), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# University of Phoenix - Online Campus
ax.annotate(x_pca_reduced_pd.index.values[1434], xy=(x_pca_reduced_pd.ix[1434,'e1'], x_pca_reduced_pd.ix[1434,'e2']), 
            xytext=(-6,6.5), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Other University of Phoenix Campuses
ax.annotate('Other University of Phoenix Campuses', xy=(x_pca_reduced_pd.ix[1499,'e1'], x_pca_reduced_pd.ix[1499,'e2']), 
            xytext=(-9.5,9.1), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Yale University
ax.annotate(x_pca_reduced_pd.index.values[180], xy=(x_pca_reduced_pd.ix[180,'e1'], x_pca_reduced_pd.ix[180,'e2']), 
            xytext=(1.2,4), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# University of Pennsylvania
ax.annotate(x_pca_reduced_pd.index.values[1076], xy=(x_pca_reduced_pd.ix[1076,'e1'], x_pca_reduced_pd.ix[1076,'e2']), 
            xytext=(-9.2,3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Columbia Centro Universitario-Caguas
ax.annotate(x_pca_reduced_pd.index.values[1379], xy=(x_pca_reduced_pd.ix[1379,'e1'], x_pca_reduced_pd.ix[1379,'e2']), 
            xytext=(-3.5,-3.6), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Caltech
ax.annotate('Caltech', xy=(x_pca_reduced_pd.ix[56,'e1'], x_pca_reduced_pd.ix[56,'e2']), 
            xytext=(4,-1.5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Inter American University of Puerto Rico
ax.annotate('IAU of Puerto Rico', xy=(x_pca_reduced_pd.ix[1391,'e1'], x_pca_reduced_pd.ix[1391,'e2']), 
            xytext=(-9.5,0), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

plt.show()


# In[9]:

# add the violin plots of each cluster across different features to come to conclusions...
# https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.violinplot.html

