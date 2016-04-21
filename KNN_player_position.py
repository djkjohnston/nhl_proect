# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:39:24 2016

purpose: test KNN on known data to predict position. This will show if (and how) descriptive the stats are
@author: danny
"""

import pandas as pd
import os
import numpy as np
import itertools

#os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project')
os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #for desktop use

#read in the data
#mean stats seem to make more logical sense because they can be applied to any player regardles of the number of games played

#player_season_sum_stats = pd.read_csv('player_season_sum_stats.csv', index_col = 0)
player_season_mean_stats = pd.read_csv('player_season_mean_stats.csv', index_col = 0)

#quick review of the data
player_season_mean_stats.head()


#identify positions and count by position
positions = set(player_season_mean_stats.position)
count = [len(player_season_mean_stats[player_season_mean_stats.position == x]) for x in positions]
zip(positions, count) #slightly more centers than i would expect

#map position to numbers
player_season_mean_stats['pos_num'] = player_season_mean_stats.position.map({'Defenseman':0, 'Center':1, 'Right Wing':2, 'Left Wing':3})

#concered that L/R wing will look similar, so grouping here
player_season_mean_stats['pos_num_simplified'] = player_season_mean_stats.position.map({'Defenseman':0, 'Center':1, 'Right Wing':2, 'Left Wing':2})

#and mapping to defense vs forward
player_season_mean_stats['pos_num_forward'] = player_season_mean_stats.position.map({'Defenseman':0, 'Center':1, 'Right Wing':1, 'Left Wing':1})


#import KNN and evaluation tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#excluding Faceoffs because they are exclusively taken by centers.
#i want to see if the KNN can work with common stats
#also removing games played b/c that may be influence by external factors
#removing timeOnIce_s as it is a sum of other features
feature_cols = ['assists', 'blocked', 'evenTimeOnIce_s', 'giveaways', 
                'goals', 'hits', 'penaltyMinutes_s', 'plusMinus', 'powerPlayAssists', 
                'powerPlayGoals', 'powerPlayTimeOnIce_s', 'shortHandedAssists', 'shortHandedGoals',
                'shortHandedTimeOnIce_s', 'shots', 'takeaways']

X = player_season_mean_stats[feature_cols]
y = player_season_mean_stats['pos_num']


#time variables are on a much larger scale than other stats. 
#Need to scalerize potential features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#test train split. Scale before splitting so scaling factors are identical
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)

print metrics.accuracy_score(y_test, y_pred_class) #pretty terrible.

#trying with pos_num_forward
y = player_season_mean_stats['pos_num_forward']

#test train split. Scale before splitting so scaling factors are identical
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)

print metrics.accuracy_score(y_test, y_pred_class) #massive jump to 96%

#trying with pos_num_simplified
y = player_season_mean_stats['pos_num_simplified']

#test train split. Scale before splitting so scaling factors are identical
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)

print metrics.accuracy_score(y_test, y_pred_class) #~10% increase over pos_num. Still not great. much worse than pos_num_forward

#want to try to optimize pos_num_simplified.
#try to find the best mix of features AND n_neighbors using cross validation
#this might take a while to run

#first, attempt cross valitation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print scores
print scores.mean() #slightly weaker than our original run


#create empty df to store results
pos_num_simplified_eval = pd.DataFrame(columns=['features','n_neighbor','accuracy'])

#testing n_neighbors 1-25
for n in range(1,26):
    #testing feature_cols
    for i in range(1, len(feature_cols)+1):
        for f in itertools.combinations(feature_cols, i):
            #set feature cols and outcome.     
            X1 = player_season_mean_stats[list(f)]
            scaler.fit(X1) #scale. is there a way to move this outside the loop?
            X1_scaled = scaler.transform(X1)
            y1 = player_season_mean_stats['pos_num_simplified']
            
            knn = KNeighborsClassifier(n_neighbors=n)
            scores = cross_val_score(knn, X1_scaled, y1, cv=10, scoring='accuracy')
            acc = scores.mean()
            
            pos_num_simplified_eval = pos_num_simplified_eval.append({'features':str(f), 
                                                                      'n_neighbor': n, 
                                                                      'accuracy': acc}, ignore_index=True)
    
pos_num_simplified_eval.shape

pos_num_simplified_eval.n_neighbor.unique #made it partially through neighbors = 1

pos_num_simplified_eval.sort_values('accuracy', ascending = False).to_csv('knn_metrics_player_pos.csv')