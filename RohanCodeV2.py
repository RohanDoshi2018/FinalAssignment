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
pd.read_csv("data.csv", low_memory=False)
labels = pd.read_cvs('lablels.csv') #### generate the labels and save..., isolate column from data and delete rows accordingly

# filter out the bad schools (rows)
data_filtered = []
for i in range(0, len(data)):
    if (data.ix[i, 'DISTANCEONLY'] == "Not distance-education only") and \
        ((data.ix[i, 'HIGHDEG'] == "Graduate degree") or (data.ix[i, 5] == "Bachelor's degree")) and \
        (data.ix[i, 'CURROPER'] == "Currently certified as operating"):
        data_filtered.append(data.ix[i])
data = pd.DataFrame(data_filtered, columns=data.columns.values)
data.drop(['Unnamed: 0', 'DISTANCEONLY', 'HIGHDEG', 'CURROPER'], axis=1)

# filter out more unecessary features (e.g. categorical features and string labels)
data.drop(['Unnamed: 0', 'DISTANCEONLY', 'HIGHDEG', 'CURROPER'], axis=1) #### replace with correct features

# normalize the data column(feature)-wise by using z-scores
stats.zscore(data, axis=1)

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
regress(linear_model.Ridge(), 'Ridge Regression')
regress(linear_model.Lasso(alpha = 0.0001), 'Lasso Regression')
regress(linear_model.ElasticNet(alpha=.0006), 'Elastic Net Regression')
regress(DecisionTreeRegressor(max_depth=10), 'Decision Tree Regression')
regress(RandomForestRegressor(n_estimators = 100, max_depth = 5, warm_start = False), 'Random Forest Regression')
regress(KNeighborsRegressor(), "KNN Regressor")
# regress(SVR(C=10, kernel='linear'), 'Linear Support Vector Regression')... takes a long time

# feature importance using random forest

# output covariance matrix heat map

# PCA plots
