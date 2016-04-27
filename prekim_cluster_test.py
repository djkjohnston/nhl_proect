# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:04:59 2016

@author: danny
"""

import pandas as pd
import os


os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project') #laptop dir
#os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #desktop dir

player_season_mean_stats = pd.read_csv('player_season_mean_stats.csv', index_col = 0)

player_season_mean_stats.head()

#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = player_season_mean_stats.drop(['games_played', 'position', 'plusMinus', 'timeOnIce_s', 'faceoffTaken', 'faceOffWins'], axis=1)
X_scaled = scaler.fit_transform(X)

#make an ugly scatter plot to get a sense of potential feature reduction
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

scatter = pd.scatter_matrix(X, figsize=(30,30), s=100)
#a few observations
#    powerplay stats seem to be moderately coorelated with overall stats. Consider removing powerplay stats except for time on ice

#run some clusters without any feature reduction
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, random_state=1)
km.fit(X_scaled)

player_season_mean_stats['cluster'] = km.labels_

player_season_mean_stats.cluster.hist()

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow', 'orange'])

scatter = pd.scatter_matrix(X, c=colors[player_season_mean_stats.cluster], figsize=(30,30), s=100)

# Save the figure (this can also be a path). As it stands now it will save in this codes directory.
plt.savefig(r"prelim_cluster_1.png")


player_season_mean_stats.columns.values

#removing some features
scaler = StandardScaler()
X = player_season_mean_stats.drop(['games_played', 'position', 'plusMinus', 'timeOnIce_s', 'faceoffTaken', 'faceOffWins', 'powerPlayAssists', 'powerPlayGoals', 'shortHandedAssists', 'shortHandedGoals'], axis=1)
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=5, random_state=1)
km.fit(X_scaled)

player_season_mean_stats['cluster'] = km.labels_

player_season_mean_stats.cluster.hist()

scatter = pd.scatter_matrix(X, c=colors[player_season_mean_stats.cluster], figsize=(30,30), s=100)
#relatively little differentiation in hits


#removing hits
scaler = StandardScaler()
X = player_season_mean_stats.drop(['games_played', 'position', 'plusMinus', 'timeOnIce_s', 'faceoffTaken', 'faceOffWins', 'powerPlayAssists', 'powerPlayGoals', 'shortHandedAssists', 'shortHandedGoals', 'hits'], axis=1)
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=5, random_state=1)
km.fit(X_scaled)

player_season_mean_stats['cluster'] = km.labels_

player_season_mean_stats.cluster.hist()

scatter = pd.scatter_matrix(X, c=colors[player_season_mean_stats.cluster], figsize=(30,30), s=100)
#this is starting to look good. The distribution of each feature is starting to look distinct by cluster.

#other items to consider
#statistically driven feature reduction
#convert time on ice columns to SHARE? does this matter with scaling
#convert goals to efficiency (goals/shots)
#test linear regression to identify +/- based on clusters
#create loop to test number of clusters?
#categorize players with <10 games?