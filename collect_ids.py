# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 10:48:52 2016

purpose: collect IDs used by NHL for season, games, teams, and players. 
source: http://www.nicetimeonice.com/api
@author: Dan Johnston
"""

import requests
import pandas as pd
import json
import os

#consistently set working directory
os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project')

#start by collecting season IDs
r_seasons = requests.get('http://www.nicetimeonice.com/api/seasons')

#r_seasons.status_code
#good status

#save for logging raw data
with open('seasons.json', 'w') as f:
    json.dump(r_seasons.json(), f)
    
#convert r_seasons to data frame for easier manipulation
df_seasons = pd.DataFrame(r_seasons.json(), columns=r_seasons.json()[0].keys())
#df_seasons.head() #confirm df built properly

#collect game IDs by season and write to files for later use
for i in df_seasons['seasonID']:
    r_games = requests.get('http://www.nicetimeonice.com/api/seasons/' + i +'/games')
    with open('season' + i + '_games.json', 'w') as f:
        json.dump(r_games.json(), f)
        
#collect and save team ids
r_teams = requests.get('http://www.nicetimeonice.com/api/teams')
with open('teams.json', 'w') as f:
    json.dump(r_teams.json(), f)