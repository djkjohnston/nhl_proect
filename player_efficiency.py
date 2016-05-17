# -*- coding: utf-8 -*-
"""
Created on Tue May 03 17:32:55 2016

@author: danny
"""

import pandas as pd
import os


os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project') #laptop dir
#os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #desktop dir

player_season_mean_stats = pd.read_csv('player_season_mean_stats.csv', index_col = 0)

player_season_mean_stats.head()

player_season_mean_stats.columns.values 

#create empty df to hold efficiency stats
player_eff = pd.DataFrame(index = player_season_mean_stats.index)

#overall stats
player_eff['timeOnIce_s'] = player_season_mean_stats['timeOnIce_s']
player_eff['shots_per_min'] = player_season_mean_stats.shots / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['goals_per_min'] = player_season_mean_stats.goals / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['goals_per_shot'] = player_season_mean_stats.goals / player_season_mean_stats.shots
player_eff['assists_per_min'] = player_season_mean_stats.assists / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['blocks_per_min'] = player_season_mean_stats.blocked / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['giveaways_per_min'] = player_season_mean_stats.giveaways / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['hits_per_min'] = player_season_mean_stats.hits / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['penaltyMinutes_per_min'] = (player_season_mean_stats.penaltyMinutes_s / 60) / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['takeaways_per_min'] = player_season_mean_stats.takeaways / (player_season_mean_stats.timeOnIce_s / 60)
player_eff['faceOff_wins_per_attempt'] = player_season_mean_stats.faceOffWins / player_season_mean_stats.faceoffTaken
#power play stats
player_eff['pp_goals_per_min'] = player_season_mean_stats.powerPlayGoals / (player_season_mean_stats.powerPlayTimeOnIce_s / 60)
player_eff['pp_assists_per_min'] = player_season_mean_stats.powerPlayAssists / (player_season_mean_stats.powerPlayTimeOnIce_s / 60)
player_eff['pp_share_of_time'] = player_season_mean_stats.powerPlayTimeOnIce_s / (player_season_mean_stats.timeOnIce_s)
#short handed stats
player_eff['sh_goals_per_min'] = player_season_mean_stats.shortHandedGoals / (player_season_mean_stats.shortHandedTimeOnIce_s / 60)
player_eff['sh_assists_per_min'] = player_season_mean_stats.shortHandedAssists / (player_season_mean_stats.shortHandedTimeOnIce_s / 60)
player_eff['sh_share_of_time'] = player_season_mean_stats.shortHandedTimeOnIce_s / (player_season_mean_stats.timeOnIce_s)

#player_eff['plusMinus_per_min'] = player_season_mean_stats.plusMinus / (player_season_mean_stats.timeOnIce / 60)
#feels more like an outcome rather than a feature. Also intend to use plusMinus in team based regression, so i dont think i want it to play a part

player_eff.shape
player_eff.head()

#recode NaN to 0
player_eff.fillna(0, inplace=True)


#scale the data
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X = player_eff
#X_scaled = pd.DataFrame(scaler.fit_transform(X))
#
##make an ugly scatter plot to get a sense of potential feature reduction
import matplotlib.pyplot as plt
#plt.rcParams['font.size'] = 14
#
#scatter = pd.scatter_matrix(X, figsize=(30,30), s=100)
#scatter = pd.scatter_matrix(X_scaled, figsize=(30,30), s=100) #try with scaled data
#
#player_eff.describe()
#
#run some clusters without any feature reduction
from sklearn.cluster import KMeans
#km = KMeans(n_clusters=5, random_state=1)
#km.fit(X_scaled)

#player_eff['cluster'] = km.labels_
#player_season_mean_stats['cluster'] = km.labels_
#
#player_eff.cluster.value_counts() #only 1 player in cluster 4
#player_eff[player_eff.cluster == 4]
#player_season_mean_stats[player_eff.cluster == 4]

import numpy as np
colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

#scatter = pd.scatter_matrix(X, c=colors[player_eff.cluster], figsize=(30,30), s=100)
#plt.gcf().savefig(r"figure_1.png")

import gc    
#test the number 
scaler = StandardScaler()
X  = player_eff
X_scaled = scaler.fit_transform(X)
    
#for i in range(2,9):
#    km = KMeans(n_clusters=i, random_state=1)
#    km.fit(X_scaled)
#    
#    player_eff['cluster'] = km.labels_
#    #player_season_mean_stats['cluster'] = km.labels_
#    #leaky memory? maybe scatter_matrix is ram hungry?
#    scatter = pd.scatter_matrix(X, c=colors[player_eff.cluster], figsize=(30,30), s=100)
#    plt.gcf().savefig(r"images\cluster_eff\all_features_n" + str(i) + ".png")
#    plt.close()
#    gc.collect()


#removing special teams stats except for share of time
scaler = StandardScaler()
X  = player_eff.drop(['pp_goals_per_min','pp_assists_per_min','sh_goals_per_min','sh_assists_per_min'], axis = 1)
#X.head()
X_scaled = scaler.fit_transform(X)
#X_temp = pd.DataFrame(X_scaled, columns=X.columns.values)
    
#scatters with new features
for b in range(2,9):
    km = KMeans(n_clusters=b, random_state=1)
    km.fit(X_scaled)
    
    #X_temp.head()
    #X_temp['cluster'] = km.labels_
    X['cluster'] = km.labels_
    #player_season_mean_stats['cluster'] = km.labels_
    #leaky memory? maybe scatter_matrix is ram hungry?
    scatter = pd.scatter_matrix(X, c=colors[X.cluster], figsize=(30,30), s=100)
    plt.gcf().savefig(r"images\cluster_eff\reduced_features1_n" + str(b) + ".png")
    plt.close()

gc.collect()

#removing faceoffs
scaler = StandardScaler()
X  = player_eff.drop(['pp_goals_per_min','pp_assists_per_min','sh_goals_per_min','sh_assists_per_min', 'faceOff_wins_per_attempt'], axis = 1)
#X.head()
X_scaled = scaler.fit_transform(X)
#X_temp = pd.DataFrame(X_scaled, columns=X.columns.values)
    
#scatters with new features
for b in range(2,9):
    km = KMeans(n_clusters=b, random_state=1)
    km.fit(X_scaled)
    
    #X_temp.head()
    #X_temp['cluster'] = km.labels_
    X['cluster'] = km.labels_
    #player_season_mean_stats['cluster'] = km.labels_
    #leaky memory? maybe scatter_matrix is ram hungry?
    scatter = pd.scatter_matrix(X, c=colors[X.cluster], figsize=(30,30), s=100)
    plt.gcf().savefig(r"images\cluster_eff\reduced_features2_n" + str(b) + ".png")
    plt.close()

gc.collect()

#removing faceoffs
scaler = StandardScaler()
X  = player_eff.drop(['pp_goals_per_min','pp_assists_per_min', 'pp_share_of_time','sh_share_of_time','sh_goals_per_min','sh_assists_per_min', 'faceOff_wins_per_attempt'], axis = 1)
#X.head()
X_scaled = scaler.fit_transform(X)
#X_temp = pd.DataFrame(X_scaled, columns=X.columns.values)
    
#scatters with new features
for b in range(2,9):
    km = KMeans(n_clusters=b, random_state=1)
    km.fit(X_scaled)
    
    #X_temp.head()
    #X_temp['cluster'] = km.labels_
    X['cluster'] = km.labels_
    #player_season_mean_stats['cluster'] = km.labels_
    #leaky memory? maybe scatter_matrix is ram hungry?
    scatter = pd.scatter_matrix(X, c=colors[X.cluster], figsize=(30,30), s=100)
    plt.gcf().savefig(r"images\cluster_eff\reduced_features3_n" + str(b) + ".png")
    plt.close()

gc.collect()