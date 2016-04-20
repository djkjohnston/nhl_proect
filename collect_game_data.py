# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 12:44:08 2016

purpose: collect individual game data for the 2013-2014 season
@author: Dan Johnston
"""

import requests
import pandas as pd
import json
import os
import time

#consistently set working directory
os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project')

#read in 2013-2014 game IDs previously collected
with open('season20132014_games.json', 'r') as f:
     json_20132014 = json.load(f)

#convert to dataframe
df_20132014 = pd.DataFrame(json_20132014, columns=json_20132014[0].keys())

df_20132014.head() #check to see if DF was made correctly
df_20132014.gameType.unique() #check gameTypes

df_20132014 = df_20132014[(df_20132014['gameType'] == 'Regular')] #subset to regular season games only. Playoffs are excluded
df_20132014.gameType.unique() #recheck gameTypes
len(df_20132014) #1230 regular season games, as expected

os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project\season20132014_games')
for i in df_20132014['gameID']:
    r_game = requests.get('http://statsapi.web.nhl.com/api/v1/game/' + i + '/feed/live')
    if r_game.status_code == 200:    
        with open('game_' + i + '.json', 'w') as f:
            json.dump(r_game.json(), f)
        time.sleep(5)