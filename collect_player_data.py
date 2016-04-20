# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:04:52 2016

purpose: loop through game data to extract player stats
@author: danny
"""

import pandas as pd
import json
import os
import numpy as np
import pandas.io.json as pij

os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project')

#read in 2013-2014 game IDs previously collected
with open('season20132014_games.json', 'r') as f:
     json_20132014 = json.load(f)
         
#convert to dataframe
df_20132014 = pd.DataFrame(json_20132014, columns=json_20132014[0].keys())
df_20132014 = df_20132014[(df_20132014['gameType'] == 'Regular')] #subset to regular season games only. Playoffs are excluded

len(df_20132014) #confirming the number of games in the regular season =1230

#create empty team columns to be filled in later
df_20132014['home_roster'] = np.NaN
df_20132014['away_roster'] = np.NaN
#retype to objects
df_20132014['home_roster'].astype('object')
df_20132014['away_roster'].astype('object')

df_20132014.head()

game_rosters = pd.DataFrame(columns=['gameID','team_name','team_ice','player', 'position', 'stats'])


for i in df_20132014.gameID:
    #read in game data
    with open('season20132014_games\game_' + i + '.json', 'r') as f:
         game = json.load(f)
    
    for j in game['liveData']['boxscore']['teams'].keys():
        for k in game['liveData']['boxscore']['teams'][j]['skaters']:
            try:    
                temp_stats = game['liveData']['boxscore']['teams'][j]['players']['ID' + str(k)]['stats']['skaterStats']
            except KeyError:
                temp_stats = np.NaN            
            
            game_rosters = game_rosters.append({'gameID':i, 
                                 'team_name': game['liveData']['boxscore']['teams'][j]['team']['name'], 
                                 'team_ice': j, 
                                 'player': str(k), 
                                 'position': game['liveData']['boxscore']['teams'][j]['players']['ID' + str(k)]['position']['name'],
                                 'stats': temp_stats}, ignore_index=True)

len(game_rosters)

game_rosters.dropna(inplace=True)

type(game_rosters.stats[0])
game_rosters.head()
game_rosters.reset_index(inplace=True) #reset index for concatting later

#game_rosters.to_csv('player_roster.csv')
#game_rosters.head(100).to_csv('player_roster_sample.csv')

#for i in game_rosters.stats:
#    print len(i.keys())

#parse stats column into seperate columns
stats_json = game_rosters.stats.to_json(orient = 'records') #seperate stats into json object
stats_df = pd.read_json(stats_json) #use read_json to create seperate columns from json dict
#stats_df.head()
#len(stats_df)

game_rosters = pd.concat([game_rosters, stats_df], axis=1)
player_game_stats = game_rosters

player_game_stats.isnull().sum()
player_game_stats.axes

#function to convert mm:ss unicode object to int variables containing the total seconds
def get_sec(s):
    l = s.split(':')
    return int(l[0]) * 60 + int(l[1])

seconder = lambda x: get_sec(x)

player_game_stats['evenTimeOnIce_s'] = player_game_stats['evenTimeOnIce'].map(seconder)
player_game_stats['penaltyMinutes_s'] = player_game_stats['penaltyMinutes']*60
player_game_stats['powerPlayTimeOnIce_s'] = player_game_stats['powerPlayTimeOnIce'].map(seconder)
player_game_stats['shortHandedTimeOnIce_s'] = player_game_stats['shortHandedTimeOnIce'].map(seconder)
player_game_stats['timeOnIce_s'] = player_game_stats['timeOnIce'].map(seconder)

player_game_stats.shape
player_game_stats.head()

#create empty df with player id as index, stats variables as columns. 
#to be used to show season cumulative sums
#Excludes faceoff percentage 
player_season_sum_stats = pd.DataFrame(index=player_game_stats.player.unique(), 
                                   columns = ['assists','blocked','evenTimeOnIce_s',
                                   'faceOffWins','faceoffTaken',
                                   'giveaways','goals','hits','penaltyMinutes_s',
                                   'plusMinus','powerPlayAssists','powerPlayGoals',
                                   'powerPlayTimeOnIce_s','shortHandedAssists',
                                   'shortHandedGoals','shortHandedTimeOnIce_s',
                                   'shots','takeaways','timeOnIce_s'])
player_season_sum_stats.shape

#loop through rows and columns and sum from 'player_game_stats'
for i in list(player_season_sum_stats.index):
    for j in player_season_sum_stats.columns.values:
        player_season_sum_stats[j][i] = player_game_stats[player_game_stats.player == i][j].sum()
        
player_season_sum_stats.head()


player_season_mean_stats = pd.DataFrame(index=player_game_stats.player.unique(), 
                                   columns = ['assists','blocked','evenTimeOnIce_s',
                                   'faceOffWins','faceoffTaken',
                                   'giveaways','goals','hits','penaltyMinutes_s',
                                   'plusMinus','powerPlayAssists','powerPlayGoals',
                                   'powerPlayTimeOnIce_s','shortHandedAssists',
                                   'shortHandedGoals','shortHandedTimeOnIce_s',
                                   'shots','takeaways','timeOnIce_s'])

#loop through rows and columns and take mean from 'player_game_stats'
for i in list(player_season_mean_stats.index):
    for j in player_season_mean_stats.columns.values:
        player_season_mean_stats[j][i] = player_game_stats[player_game_stats.player == i][j].mean()
        
player_season_mean_stats.head()

#add games_played column to both summary dfs
player_season_sum_stats['games_played'] = np.NaN
player_season_mean_stats['games_played'] = np.NaN

#both DFs have the same indexes in the same order, so i can update both with one loop
for i in player_season_sum_stats.index:
    x = len(player_game_stats[player_game_stats.player == i])
    player_season_sum_stats['games_played'][i] = x
    player_season_mean_stats['games_played'][i] = x
#results in warnings, but seems to work?
player_season_mean_stats.head()

#add games played to both summary dfs



#save the file for later use
player_game_stats.to_csv('player_game_stats.csv')
player_season_sum_stats.to_csv('player_season_sum_stats.csv')
player_season_mean_stats.to_csv('player_season_mean_stats.csv')