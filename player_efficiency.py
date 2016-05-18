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
    #scatter = pd.scatter_matrix(X, c=colors[X.cluster], figsize=(30,30), s=100)
    #plt.gcf().savefig(r"images\cluster_eff\reduced_features1_n" + str(b) + ".png")
    #plt.close()
    X.cluster.hist()

gc.collect()

#removing special teams and faceoffs
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

#removing special teams, faceoffs, and special team time on ice
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

#most clustering tests end up with 1 or more clusters with <5 players. I need to figure out who those players are

player_cluster = pd.DataFrame(index = player_season_mean_stats.index)

scaler = StandardScaler()
X  = player_eff.drop(['pp_goals_per_min','pp_assists_per_min', 'pp_share_of_time','sh_share_of_time','sh_goals_per_min','sh_assists_per_min', 'faceOff_wins_per_attempt'], axis = 1)
X_scaled = scaler.fit_transform(X)
    
for b in range(2,9):
    km = KMeans(n_clusters=b, random_state=1)
    km.fit(X_scaled)
    player_cluster['cluster_' + str(b)] = km.labels_

player_cluster.cluster_2.hist() #fine
player_cluster.cluster_3.hist() #fine
player_cluster.cluster_4.hist() #cluster 3
player_cluster.cluster_5.hist() #cluster 4
player_cluster.cluster_6.hist() #cluster 4
player_cluster.cluster_7.hist() #clusters 2, 3
player_cluster.cluster_8.hist() #clusters 4, 5

cluster_4_concerns = player_cluster[player_cluster.cluster_4 == 3].index
cluster_5_concerns = player_cluster[player_cluster.cluster_5 == 4].index
cluster_6_concerns = player_cluster[player_cluster.cluster_6 == 4].index
cluster_7_concerns = player_cluster[(player_cluster.cluster_7 == 2) | (player_cluster.cluster_7 == 3)].index
cluster_8_concerns = player_cluster[(player_cluster.cluster_8 == 4) | (player_cluster.cluster_8 == 5)].index

player_season_mean_stats.ix[cluster_4_concerns] #players with few games played
player_season_mean_stats.ix[cluster_5_concerns] #players with few games played
player_season_mean_stats.ix[cluster_6_concerns] #players with few games played
player_season_mean_stats.ix[cluster_7_concerns] #largely players with few games played, some players with more games
player_season_mean_stats.ix[cluster_8_concerns] #largely players with few games played, some players with more games


#it looks like players with few games are driving the outliers. Lets remove players with fewer than 5 games and try reclustering.
player_season_mean_stats.games_played.value_counts(sort=False)

player_season_mean_stats_trimmed = player_season_mean_stats[player_season_mean_stats.games_played >= 5]

len(player_season_mean_stats)
len(player_season_mean_stats_trimmed) #drops 81 players

#create empty df to hold efficiency stats
player_eff_trimmed = pd.DataFrame(index = player_season_mean_stats_trimmed.index)

#overall stats
player_eff_trimmed['timeOnIce_s'] = player_season_mean_stats_trimmed['timeOnIce_s']
player_eff_trimmed['shots_per_min'] = player_season_mean_stats_trimmed.shots / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['goals_per_min'] = player_season_mean_stats_trimmed.goals / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['goals_per_shot'] = player_season_mean_stats_trimmed.goals / player_season_mean_stats_trimmed.shots
player_eff_trimmed['assists_per_min'] = player_season_mean_stats_trimmed.assists / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['blocks_per_min'] = player_season_mean_stats_trimmed.blocked / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['giveaways_per_min'] = player_season_mean_stats_trimmed.giveaways / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['hits_per_min'] = player_season_mean_stats_trimmed.hits / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['penaltyMinutes_per_min'] = (player_season_mean_stats_trimmed.penaltyMinutes_s / 60) / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['takeaways_per_min'] = player_season_mean_stats_trimmed.takeaways / (player_season_mean_stats_trimmed.timeOnIce_s / 60)
player_eff_trimmed['faceOff_wins_per_attempt'] = player_season_mean_stats_trimmed.faceOffWins / player_season_mean_stats_trimmed.faceoffTaken
#power play stats
player_eff_trimmed['pp_goals_per_min'] = player_season_mean_stats_trimmed.powerPlayGoals / (player_season_mean_stats_trimmed.powerPlayTimeOnIce_s / 60)
player_eff_trimmed['pp_assists_per_min'] = player_season_mean_stats_trimmed.powerPlayAssists / (player_season_mean_stats_trimmed.powerPlayTimeOnIce_s / 60)
player_eff_trimmed['pp_share_of_time'] = player_season_mean_stats_trimmed.powerPlayTimeOnIce_s / (player_season_mean_stats_trimmed.timeOnIce_s)
#short handed stats
player_eff_trimmed['sh_goals_per_min'] = player_season_mean_stats_trimmed.shortHandedGoals / (player_season_mean_stats_trimmed.shortHandedTimeOnIce_s / 60)
player_eff_trimmed['sh_assists_per_min'] = player_season_mean_stats_trimmed.shortHandedAssists / (player_season_mean_stats_trimmed.shortHandedTimeOnIce_s / 60)
player_eff_trimmed['sh_share_of_time'] = player_season_mean_stats_trimmed.shortHandedTimeOnIce_s / (player_season_mean_stats_trimmed.timeOnIce_s)


player_eff_trimmed.fillna(0, inplace=True)
player_eff_trimmed.head()
#and now for clustering
scaler = StandardScaler()
X  = player_eff_trimmed.drop(['pp_goals_per_min','pp_assists_per_min', 'pp_share_of_time','sh_share_of_time','sh_goals_per_min','sh_assists_per_min', 'faceOff_wins_per_attempt'], axis = 1)
X_scaled = scaler.fit_transform(X)

#for b in range(2,9):
#    km = KMeans(n_clusters=b, random_state=1)
#    km.fit(X_scaled)
#    
#    X['cluster'] = km.labels_
#    scatter = pd.scatter_matrix(X, c=colors[X.cluster], figsize=(30,30), s=100)
#    plt.gcf().savefig(r"images\cluster_eff_trimmed\reduced_features3_n" + str(b) + ".png")
#    plt.close()
#
#gc.collect()
#things are looking better, but i want to...

#take a look at the hists
player_cluster_trimmed = pd.DataFrame(index = player_season_mean_stats_trimmed.index)

scaler = StandardScaler()
X  = player_eff_trimmed.drop(['pp_goals_per_min','pp_assists_per_min', 'pp_share_of_time','sh_share_of_time','sh_goals_per_min','sh_assists_per_min', 'faceOff_wins_per_attempt'], axis = 1)
X_scaled = scaler.fit_transform(X)


for b in range(2,9):
    km = KMeans(n_clusters=b, random_state=1)
    km.fit(X_scaled)
    player_cluster_trimmed['cluster_' + str(b)] = km.labels_

player_cluster_trimmed.cluster_2.hist() #fine
player_cluster_trimmed.cluster_3.hist() #fine
player_cluster_trimmed.cluster_4.hist() #cluster 1 looks smaller, but it still has ~40+ players
player_cluster_trimmed.cluster_5.hist() #cluster 4 is smaller, but has ~30+ players
player_cluster_trimmed.cluster_6.hist() #cluster 1 looks smaller, but it still has ~30+ players
player_cluster_trimmed.cluster_7.hist() #clusters 2, 3
player_cluster_trimmed.cluster_8.hist() #clusters 2 is the smallest with ~25 players

#these clusters feel much better, but let's look at the feature selection with a bit more rigor
player_eff_trimmed.corr().to_csv('player_eff_correlation.csv')
pd.scatter_matrix(player_eff_trimmed, figsize=(30,30), s=100)
#goals_per_shot and goals_per_min are highly correlated. Stick with goals_per_shot
#some stats like shots and blocks are moderately negatively correlated... difference between Dfense and Ofense?
#PP and SH stats seem less correlated than I expected. I think i'll try to keep share of time, but still want to remove the other stats. SH and PP stats only have goals and assists, nothing about blocks/hits. etc


#test different cluster/feature combinations using silhouette coefficient as the metric
features = ['timeOnIce_s', 'shots_per_min', 'goals_per_shot', 'assists_per_min', 
            'blocks_per_min', 'giveaways_per_min', 'hits_per_min', 
            'penaltyMinutes_per_min', 'takeaways_per_min', 'faceOff_wins_per_attempt',  
            'pp_share_of_time', 'sh_share_of_time']

from sklearn import metrics 
import multiprocessing as mp
import itertools

combination_list = [] # create a list to store the combinations

#testing n_neighbors 1-25
for n in range(2,19):#18 non-goalie players per team
    #testing feature_cols
    for i in range(1, len(features)+1): 
        for f in itertools.combinations(features, i):
            D = {'clusters': n, 'features': f}
            combination_list.append(D) # append this combination to the list        

len(combination_list)        

def player_eff_trimmed_cluster(D):
    #set feature cols and outcome.
    X = player_eff_trimmed[list(D['features'])]
    scaler.fit(X) #scale. is there a way to move this outside the loop?
    X_scaled = scaler.transform(X)
    km = KMeans(n_clusters=D['clusters'], random_state=1)
    km.fit(X_scaled)
    silhouette_score = metrics.silhouette_score(X_scaled, km.labels_)
    
    return {'features':str(D['features']), 'n_clusters': D['clusters'], 'silhouette': silhouette_score}
#should have added cluster sizes
scaler = StandardScaler()
templist = [player_eff_trimmed_cluster(combination_list[0]), player_eff_trimmed_cluster(combination_list[-1])]

pool = mp.Pool()
mp.cpu_count()


#silhouette_df = pd.DataFrame(pool.map(player_eff_trimmed_cluster, combination_list[0:10]))

silhouette_df = []
silhouette_df = [player_eff_trimmed_cluster(i) for i in combination_list]

silhouette_df = pd.DataFrame(silhouette_df)

silhouette_df.to_csv('silhouette_cluster_tests.csv')
player_eff_trimmed.to_csv('player_eff_trimmed.csv')