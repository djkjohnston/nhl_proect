# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:56:48 2016

@author: danny
"""

import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project') #laptop dir
#os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #desktop dir

#read in silhouette results and clean features column. Should look for better writing in future
silhouette_df = pd.read_csv('silhouette_cluster_tests.csv')
silhouette_df['features'] = silhouette_df['features'].map(lambda x: x.lstrip('(').rstrip(',)'))
silhouette_df['features'] = silhouette_df['features'].str.replace('\'', '')
silhouette_df.tail()
#I see that some clusters use very few features. I am curious about how large the resulting clusters are
#create top 1000
silhouette_1000 = silhouette_df.sort_values(by='silhouette', ascending=False).head(1000)
silhouette_1000.head()

player_eff_trimmed = pd.read_csv('player_eff_trimmed.csv', index_col=0)
player_eff_trimmed.head()

silhouette_1000_minmax = pd.DataFrame(columns=['min','max'], index=silhouette_1000.index)

scaler = StandardScaler()
#fill in the min and max size of clusters
for i in silhouette_1000_minmax.index:
    X  = player_eff_trimmed[silhouette_1000['features'][i].split(', ')]
    X_scaled = scaler.fit_transform(X)
    km = KMeans(silhouette_1000['n_clusters'][i], random_state=1)
    km.fit(X_scaled)
    silhouette_1000_minmax['min'][i] = np.bincount(km.labels_).min()
    silhouette_1000_minmax['max'][i] = np.bincount(km.labels_).max()

silhouette_1000_minmax.head()

silhouette_1000 = pd.concat([silhouette_1000, silhouette_1000_minmax], axis = 1)

#remove entries where min cluster size < 30 (there are 30 teams in the league)
#also remove entries where there are fewer than 5 clusters
silhouette_top = silhouette_1000[(silhouette_1000['min'] >= 30)]
len(silhouette_top) #dropped more than half
silhouette_top.head(50)
silhouette_top.sort_values(by='n_clusters', ascending=False).head(50)

#decided to exclude sh and pp fields from clustering. I reran the clusters and load them here:
silhouette_df2 = pd.read_csv('silhouette_cluster_tests2.csv')
silhouette_df2['features'] = silhouette_df2['features'].map(lambda x: x.lstrip('(').rstrip(',)'))
silhouette_df2['features'] = silhouette_df2['features'].str.replace('\'', '')
silhouette_df2['n_features'] = [i.count(',')+1 for i in silhouette_df2['features']]
silhouette_df2.head()
silhouette_df2.tail()

len(silhouette_df2[(silhouette_df2['n_features'] >= 3) & (silhouette_df2['smallest_cluster'] >= 30) & (silhouette_df2['n_clusters'] >= 5)].sort_values(by='silhouette', ascending=False))
silhouette_df2[(silhouette_df2['n_features'] >= 3) & (silhouette_df2['smallest_cluster'] >= 30) & (silhouette_df2['n_clusters'] >= 5)].sort_values(by='silhouette', ascending=False).head(20)

#what i want to do now
#1. recalculate and match clusters to player/game data
#2. aggretate player/game data to show the count of players in each cluster per game
#3. match team +/- by game to #2
#4. run a linear model using clusters to predict +/-


player_game = pd.read_csv('player_game_stats.csv', index_col='index', usecols=['index', 'gameID', 'team_name', 'team_ice', 'player'])
player_game.head()
scaler = StandardScaler()
#fill in the min and max size of clusters
for i in silhouette_df2.index[0]:
    X  = player_eff_trimmed[silhouette_df2['features'][0].split(', ')]
    X_scaled = scaler.fit_transform(X)
    km = KMeans(silhouette_df2['n_clusters'][0], random_state=1)
    km.fit(X_scaled)
    X['cluster'] = list(km.labels_) #causes a warning but doesnt break. Need to investigate
    
X.head()