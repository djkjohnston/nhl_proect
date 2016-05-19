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
import json

#os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project') #laptop dir
os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #desktop dir

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
#1. match team +/- by game 
#2. recalculate and match clusters to player/game data
#3. aggretate player/game data to show the count of players in each cluster per game, and match to #2
#4. run a linear model using clusters to predict +/-

########################################################
#1. match team +/- by game 
player_game = pd.read_csv('player_game_stats.csv', index_col='index', usecols=['index', 'gameID', 'team_name', 'team_ice', 'player'])
player_game_scores = player_game.drop(['player'], axis=1).drop_duplicates()
player_game_scores['goals'] = ""
player_game_scores.reset_index(inplace=True)

#pull in list of games
with open('season20132014_games.json', 'r') as f:
     json_20132014 = json.load(f)
         
#convert to dataframe
df_20132014 = pd.DataFrame(json_20132014, columns=json_20132014[0].keys())
df_20132014 = df_20132014[(df_20132014['gameType'] == 'Regular')] #subset to regular season games only. Playoffs are excluded
#df_20132014.head()

#bring in goals by game by team
for i in df_20132014.gameID:
    #read in game data
    with open('season20132014_games\game_' + i + '.json', 'r') as f:
         game = json.load(f)
    
    for j in game['liveData']['linescore']['teams'].keys():
        player_game_scores.set_value((player_game_scores['gameID'] == int(i)) & (player_game_scores['team_ice']==j), 'goals', game['liveData']['linescore']['teams'][j]['goals'])

#calculate plus/minus
player_game_scores['plus_minus'] = "" #start with an empty cell
for i in player_game_scores.index:
    #home/away games are always paired and ordered by home than away. This means I can use even/odd of the index to set the operation for calculating +/-    
    if i % 2 == 0:
        player_game_scores.set_value(i, 'plus_minus', player_game_scores['goals'][i] - player_game_scores['goals'][i+1])
    else:
        player_game_scores.set_value(i, 'plus_minus', player_game_scores['goals'][i] - player_game_scores['goals'][i-1])

#resort and indexing to make matching in step 3 easier    
player_game_scores = player_game_scores.sort_values(by=['gameID','team_ice']).reset_index()
#########################################################
#2. recalculate and match clusters to player/game data
scaler = StandardScaler()

X  = player_eff_trimmed[silhouette_df2['features'][0].split(', ')]
X_scaled = scaler.fit_transform(X)
km = KMeans(silhouette_df2['n_clusters'][0], random_state=1)
km.fit(X_scaled)
temp_df = pd.DataFrame(km.labels_, index=X.index, columns =['cluster'])
player_game['cluster']=player_game.player.map(temp_df.cluster, na_action ='ignore') #match to player_game
player_game.head()

#a couple visual checks to make sure assignments are consistent for all player instances
#player_game[player_game['player'] == 8475153]
#player_game[player_game['player'] == 8474498]
#player_game[player_game['player'] == 8476522] #a player that was explicitly dropped due to low number of games

###############################################################
#3. aggretate player/game data to show the count of players in each cluster per game, and match to #2
#add team plus_minuses to player_data 
player_game.head()

temp_dummies = pd.get_dummies(player_game.cluster, prefix='cluster') #convert clusters into dummy variables. note excluding any variables because there are respondents with unassigned clusters
temp = pd.concat([player_game.drop(['cluster', 'player'], axis=1), temp_dummies], axis=1)

temp = temp.groupby(['gameID','team_name','team_ice']).sum().reset_index()
#resort and indexing to make matching to player_game_scores easier
temp = temp.sort_values(by=['gameID','team_ice']).reset_index()

temp['plus_minus'] = player_game_scores['plus_minus']
temp.head()

#################################################################
#4. run a linear model using clusters to predict +/-
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression

feature_cols = [col for col in temp.columns if 'cluster' in col] #find all columns including 'cluster' in the name. Makes it more scalable

X = temp[feature_cols]
y = temp['plus_minus']

linreg = LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')

np.mean(np.sqrt(abs(scores)))

###################################################################
#lets see if i can functionalize this to test a few clustering scenarios

scaler = StandardScaler()

def test_clusters(i):
    #run the clustering
    X  = player_eff_trimmed[silhouette_df2['features'][i].split(', ')]
    X_scaled = scaler.fit_transform(X)
    km = KMeans(silhouette_df2['n_clusters'][i], random_state=1)
    km.fit(X_scaled)
    #store clusters in a data fra
    temp_clusters = pd.DataFrame(km.labels_, index=X.index, columns =['cluster'])
    temp_player = player_game
    temp_player['cluster']=temp_player.player.map(temp_clusters.cluster, na_action ='ignore') #match to player_game
    temp_dummy = pd.get_dummies(temp_player.cluster, prefix='cluster') #convert clusters into dummy variables. not excluding any variables because there are respondents with unassigned clusters

    temp_player = pd.concat([player_game.drop([ 'player'], axis=1), temp_dummy], axis=1)

    temp_player = temp_player.groupby(['gameID','team_name','team_ice']).sum().reset_index()
    #resort and indexing to make matching to player_game_scores easier
    temp_player = temp_player.sort_values(by=['gameID','team_ice']).reset_index()
    temp_player['plus_minus'] = player_game_scores['plus_minus']
    
    feature_cols = [col for col in temp_player.columns if 'cluster' in col] #find all columns including 'cluster' in the name. Makes it more scalable

    X = temp_player[feature_cols]
    y = temp_player['plus_minus']
    
    linreg = LinearRegression()
    scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')
    
    return np.mean(np.sqrt(abs(scores)))
    
silhouette_500 = silhouette_df2[(silhouette_df2['n_features'] >= 3) & (silhouette_df2['smallest_cluster'] >= 30) & (silhouette_df2['n_clusters'] >= 5)].sort_values(by='silhouette', ascending=False).head(500)

silhouette_500['rsme'] = [test_clusters(i) for i in silhouette_500.index]

silhouette_500.sort_values(by='rsme', ascending=False).head()
player_game_scores.plus_minus.describe(include = 'list-like')
np.percentile(player_game_scores.plus_minus, [0,25,50,75,100])

silhouette_500.to_csv('top_500_clusters_LM_results.csv')