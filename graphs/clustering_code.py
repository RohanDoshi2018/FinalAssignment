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

data = pd.read_csv("../x_named_z.csv", low_memory=False, index_col=0)
labels = pd.read_csv('../y_named.csv', header=None) 

x = data
x = x.drop('UNITID', axis=1)

y = pd.read_csv('../y_named.csv', low_memory=False, header=None)
schools = y.ix[:,0]
y.index = schools
list_schools = schools.tolist()
y.index = list_schools
y = y.drop(0,axis=1)

pca = decomposition.PCA(n_components=2)
pca.fit(x)
x_pca_reduced = pca.fit_transform(x)
x_pca_reduced_pd = pd.DataFrame(data=x_pca_reduced, index=list_schools, columns=['e1', 'e2'])  # 1st row as the column names

## TECH

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax_tech = fig.add_subplot(111)
ax_tech.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# MIT
ax_tech.annotate(x_pca_reduced_pd.index.values[524], xy=(x_pca_reduced_pd.ix[524,'e1'],x_pca_reduced_pd.ix[524,'e2']), 
            xytext=(-3.5,-3.2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Caltech
ax_tech.annotate('Caltech', xy=(x_pca_reduced_pd.ix[56,'e1'], x_pca_reduced_pd.ix[56,'e2']), 
            xytext=(4,-1.5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Virginia Tech
ax_tech.annotate('Virginia Tech', xy=(x_pca_reduced_pd.ix[1300,'e1'],x_pca_reduced_pd.ix[1300,'e2']), 
            xytext=(-8.5,-2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Wentworth Institute of Technology
ax_tech.annotate(x_pca_reduced_pd.index.values[545], xy=(x_pca_reduced_pd.ix[545,'e1'],x_pca_reduced_pd.ix[545,'e2']), 
            xytext=(-9.5,8), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Georgia Tech
ax_tech.annotate('Georgia Tech', xy=(x_pca_reduced_pd.ix[247,'e1'],x_pca_reduced_pd.ix[247,'e2']), 
            xytext=(0,6), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Lawrence Technological University
ax_tech.annotate(x_pca_reduced_pd.index.values[568], xy=(x_pca_reduced_pd.ix[568,'e1'],x_pca_reduced_pd.ix[568,'e2']), 
            xytext=(-9.2,2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Michigan Technological University
ax_tech.annotate(x_pca_reduced_pd.index.values[574], xy=(x_pca_reduced_pd.ix[574,'e1'],x_pca_reduced_pd.ix[574,'e2']), 
            xytext=(-9.9,3.5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# NJIT
ax_tech.annotate('NJIT', xy=(x_pca_reduced_pd.ix[725,'e1'],x_pca_reduced_pd.ix[725,'e2']), 
            xytext=(-8,.75), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

plt.show()

## MUSIC

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax_music = fig.add_subplot(111)
ax_music.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# McNally
ax_music.annotate(x_pca_reduced_pd.index.values[1431], xy=(x_pca_reduced_pd.ix[1431,'e1'],x_pca_reduced_pd.ix[1431,'e2']), 
            xytext=(-1.5,-3.2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# MSM
ax_music.annotate(x_pca_reduced_pd.index.values[796], xy=(x_pca_reduced_pd.ix[796,'e1'], x_pca_reduced_pd.ix[796,'e2']), 
            xytext=(-1,-2.3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# NEC
ax_music.annotate(x_pca_reduced_pd.index.values[530], xy=(x_pca_reduced_pd.ix[530,'e1'], x_pca_reduced_pd.ix[530,'e2']), 
            xytext=(-4,10), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Juilliard
ax_music.annotate(x_pca_reduced_pd.index.values[787], xy=(x_pca_reduced_pd.ix[787,'e1'],x_pca_reduced_pd.ix[787,'e2']), 
            xytext=(-9.5,8), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Oberlin
ax_music.annotate(x_pca_reduced_pd.index.values[951], xy=(x_pca_reduced_pd.ix[951,'e1'],x_pca_reduced_pd.ix[951,'e2']), 
            xytext=(-9,-3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Rochester (Eastman)
ax_music.annotate(x_pca_reduced_pd.index.values[820], xy=(x_pca_reduced_pd.ix[820,'e1'],x_pca_reduced_pd.ix[820,'e2']), 
            xytext=(-9.2,2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Bard
ax_music.annotate(x_pca_reduced_pd.index.values[752], xy=(x_pca_reduced_pd.ix[752,'e1'],x_pca_reduced_pd.ix[752,'e2']), 
            xytext=(-9,0), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

plt.show()