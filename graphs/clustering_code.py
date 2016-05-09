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

## LIBERAL ARTS

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax_la = fig.add_subplot(111)
ax_la.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# Williams
ax_la.annotate(x_pca_reduced_pd.index.values[549], xy=(x_pca_reduced_pd.ix[549,'e1'],x_pca_reduced_pd.ix[549,'e2']), 
            xytext=(2,-2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Middlebury
ax_la.annotate(x_pca_reduced_pd.index.values[1271], xy=(x_pca_reduced_pd.ix[1271,'e1'], x_pca_reduced_pd.ix[1271,'e2']), 
            xytext=(0,-3.5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Bard
ax_la.annotate(x_pca_reduced_pd.index.values[752], xy=(x_pca_reduced_pd.ix[752,'e1'], x_pca_reduced_pd.ix[752,'e2']), 
            xytext=(-9.5,0), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Claremont McKenna
ax_la.annotate(x_pca_reduced_pd.index.values[89], xy=(x_pca_reduced_pd.ix[89,'e1'],x_pca_reduced_pd.ix[89,'e2']), 
            xytext=(-4,10), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Wesley
ax_la.annotate(x_pca_reduced_pd.index.values[184], xy=(x_pca_reduced_pd.ix[184,'e1'],x_pca_reduced_pd.ix[184,'e2']), 
            xytext=(-9.5,8), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Rhodes
ax_la.annotate(x_pca_reduced_pd.index.values[1169], xy=(x_pca_reduced_pd.ix[1169,'e1'],x_pca_reduced_pd.ix[1169,'e2']), 
            xytext=(-9,-3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Southern New Hampshire
ax_la.annotate(x_pca_reduced_pd.index.values[705], xy=(x_pca_reduced_pd.ix[705,'e1'],x_pca_reduced_pd.ix[705,'e2']), 
            xytext=(-9.5,2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))


plt.show()

## PRIVATE

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax_priv = fig.add_subplot(111)
ax_priv.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# Princeton
ax_priv.annotate(x_pca_reduced_pd.index.values[726], xy=(x_pca_reduced_pd.ix[726,'e1'],x_pca_reduced_pd.ix[726,'e2']), 
            xytext=(0.75,-2.2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Scripps
ax_priv.annotate(x_pca_reduced_pd.index.values[132], xy=(x_pca_reduced_pd.ix[132,'e1'], x_pca_reduced_pd.ix[132,'e2']), 
            xytext=(2,-1), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Franklin Pierce
ax_priv.annotate('Franlin Pierce', xy=(x_pca_reduced_pd.ix[703,'e1'], x_pca_reduced_pd.ix[703,'e2']), 
            xytext=(-9.5,0), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Grand Canyon
ax_priv.annotate(x_pca_reduced_pd.index.values[28], xy=(x_pca_reduced_pd.ix[28,'e1'],x_pca_reduced_pd.ix[28,'e2']), 
            xytext=(-4,7), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Post
ax_priv.annotate(x_pca_reduced_pd.index.values[171], xy=(x_pca_reduced_pd.ix[171,'e1'],x_pca_reduced_pd.ix[171,'e2']), 
            xytext=(-9.5,8), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Universidad Del Este
ax_priv.annotate(x_pca_reduced_pd.index.values[1400], xy=(x_pca_reduced_pd.ix[1400,'e1'],x_pca_reduced_pd.ix[1400,'e2']), 
            xytext=(-2,-3.5), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Bucknell University
ax_priv.annotate(x_pca_reduced_pd.index.values[1021], xy=(x_pca_reduced_pd.ix[1021,'e1'],x_pca_reduced_pd.ix[1021,'e2']), 
            xytext=(-9.5,2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Occidental
ax_priv.annotate(x_pca_reduced_pd.index.values[112], xy=(x_pca_reduced_pd.ix[112,'e1'],x_pca_reduced_pd.ix[112,'e2']), 
            xytext=(2,8), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Drexel
ax_priv.annotate(x_pca_reduced_pd.index.values[1033], xy=(x_pca_reduced_pd.ix[1033,'e1'],x_pca_reduced_pd.ix[1033,'e2']), 
            xytext=(-9,-2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Davenport
ax_priv.annotate(x_pca_reduced_pd.index.values[560], xy=(x_pca_reduced_pd.ix[560,'e1'],x_pca_reduced_pd.ix[560,'e2']), 
            xytext=(-9.5,3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

plt.show()

## TRADE

clusters = KMeans().fit_predict(x) #n_clusters=5
fig = plt.figure()
ax_trade = fig.add_subplot(111)
ax_trade.scatter(x_pca_reduced_pd.ix[:, 0], x_pca_reduced_pd.ix[:, 1], c=clusters, s=30)

# Mass MA
ax_trade.annotate(x_pca_reduced_pd.index.values[525], xy=(x_pca_reduced_pd.ix[525,'e1'],x_pca_reduced_pd.ix[525,'e2']), 
            xytext=(-6.5,9.2), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# CMA
ax_trade.annotate(x_pca_reduced_pd.index.values[83], xy=(x_pca_reduced_pd.ix[83,'e1'], x_pca_reduced_pd.ix[83,'e2']), 
            xytext=(-1,10), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Maine MA
ax_trade.annotate(x_pca_reduced_pd.index.values[460], xy=(x_pca_reduced_pd.ix[460,'e1'], x_pca_reduced_pd.ix[460,'e2']), 
            xytext=(-9.5,6), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# SUNY MC
ax_trade.annotate(x_pca_reduced_pd.index.values[854], xy=(x_pca_reduced_pd.ix[854,'e1'],x_pca_reduced_pd.ix[854,'e2']), 
            xytext=(-8,8), color='purple', arrowprops=dict(facecolor='red', shrink=0.05))

# Calvary Bible
ax_trade.annotate('Calvary Bible College', xy=(x_pca_reduced_pd.ix[634,'e1'],x_pca_reduced_pd.ix[634,'e2']), 
            xytext=(-9.5,0), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Grace 
ax_trade.annotate('Grace Theological Seminary', xy=(x_pca_reduced_pd.ix[336,'e1'],x_pca_reduced_pd.ix[336,'e2']), 
            xytext=(-3,-3.3), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

# Faith
ax_trade.annotate('Faith Baptist Bible College', xy=(x_pca_reduced_pd.ix[374,'e1'],x_pca_reduced_pd.ix[374,'e2']), 
            xytext=(-9.5,2), color='purple', arrowprops=dict(facecolor='yellow', shrink=0.05))

plt.show()