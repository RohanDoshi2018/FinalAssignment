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
%matplotlib inline


# read in data and labels
d = pd.read_csv("data.csv", low_memory=False)
# labels = pd.read_cvs('lablels.csv') #### generate the labels and save..., isolate column from data and delete rows accordingly

# normalize the data column(feature)-wise by using z-scores
d = stats.zscore(d, axis=1)

def plotPred(y_predict, y_test, name):
    plt.figure()
    plt.scatter(y_test, y_predict)
    t = np.arange(0, max(np.amax(y_predict),np.amax(y_test)), 1)
    plt.plot(t,t)
    plt.xlabel('Actual incomes')
    plt.ylabel('Predicted incomes')
    plt.title(name)
    plt.show()
    print("RMSE:", rmse(y_test, y_predict))    
    print("r2:", r2_score(y_test, y_predict))

def plotResiduals(y_predict, y_test, name):
    plt.figure()
    plt.hist(y_predict-y_test)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(name + ' Residuals')
    plt.show()

# http://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def listTenLargestResiduals(y_predict, y_test) 
	residuals = list(y_predict-y_test).sort()
	for x in range(10):
		print(residuals[i])

#### gridsearch hyperparameter optimization
# evaluate with 3-fold finetuning 

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

# fit and test the regressor, output graph and evaluation statistics
def regress(model, name):
    reg = model
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    plotPred(y_predict, y_test, name)
    plotResiduals(y_predict, y_test, name)
    listTenLargestResiduals(y_predict, y_test)

# test different regression models - still need to finetune
regress(linear_model.LinearRegression(), 'Ordinary Least Squares Regression')
# regress(linear_model.Ridge(), 'Ridge Regression')
# regress(linear_model.Lasso(), 'Lasso Regression')
# regress(linear_model.ElasticNet(), 'Elastic Net Regression')
# regress(DecisionTreeRegressor(max_depth=10), 'Decision Tree Regression')
# regress(RandomForestRegressor(n_estimators = 100, max_depth = 5, warm_start = False), 'Random Forest Regression')
# regress(KNeighborsRegressor(), "KNN Regressor")
# regress(SVR(C=10, kernel='linear'), 'Linear Support Vector Regression')... takes a long time

# feature importance using random forest

# output covariance matrix heat map

# PCA plots
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn import decomposition


pca = decomposition.PCA(n_components=3)
pca.fit(d)
pca_d = pca.transform(d)
e1 = pca_d[:,0]
e2 = pca_d[:,1]
plt.plot(e1, e2)
plot.show()



