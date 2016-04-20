# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:52:08 2016

purpose: explore game level data
@author: danny
"""

import pandas as pd
import json
import os
import time
import sys
import numpy as np

#consistently set working directory
os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project\season20132014_games')

#read in game_2013020001.json
with open('game_2013020001.json', 'r') as f:
     game_2013020001 = json.load(f)
     
game_2013020001.keys()

game_2013020001['copyright'] # not helpful
game_2013020001['gameData'] # probably what i need
game_2013020001['link'] # not helpful
game_2013020001['liveData'] # looks like game summary. Potentially most useful
game_2013020001['gamePk'] # not helpful
game_2013020001['metaData'] #  describes details of the API call

liveData_2013020001 = game_2013020001['liveData']

type(liveData_2013020001)

liveData_2013020001.keys()

liveData_2013020001['linescore'] #game summary data at the team level
liveData_2013020001['boxscore'] #looks like player summary stats. could be useful
liveData_2013020001['plays'] #play by play data, may be too difficult to parse
liveData_2013020001['decisions'] #game awards 1st-3rd stars, etc. Not important

boxscore_201302001 = liveData_2013020001['boxscore']

boxscore_201302001.keys()

boxscore_201302001['officials'] #referees and linesman
boxscore_201302001['teams'] #teams data. May have player data nested

game_2013020001['liveData']['boxscore']['teams'] == boxscore_201302001['teams'] #testing selecting data from nested json

teambox_2013020001 = boxscore_201302001['teams']

teambox_2013020001.keys()

game_2013020001['liveData']['boxscore']['teams']['home']['players'].keys()

for i in game_2013020001['liveData']['boxscore']['teams']['home']['players'].keys():
    try:    
        temp_stats = game_2013020001['liveData']['boxscore']['teams']['home']['players'][i]['stats']['skaterStats']
    except KeyError:
        temp_stats = np.NaN
    
game_2013020001['liveData']['boxscore']['teams']['home']['players']['ID8474189']['stats']['skaterStats']