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

os.chdir(r'C:\Users\danny\GA_DataScience\nhl_project')
#os.chdir(r'C:\Users\Dan-PC\GA_DataScience\nhl_project') #for desktop use

#read in the data
#mean stats seem to make more logical sense because they can be applied to any player regardles of the number of games played

#player_season_sum_stats = pd.read_csv('player_season_sum_stats.csv', index_col = 0)
player_season_mean_stats = pd.read_csv('player_season_mean_stats.csv', index_col = 0)

#quick review of the data
player_season_mean_stats.head()
len(player_season_mean_stats)

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

#from multiprocessing import Pool 
import multiprocessing

pool = multiprocessing.Pool()
multiprocessing.cpu_count()


#create empty df to store results
pos_num_simplified_eval = pd.DataFrame(columns=['features','n_neighbor','accuracy'])

combination_list = [] # create a list to store the combinations

#testing n_neighbors 1-25
for n in range(1,3):#26):
    #testing feature_cols
    for i in range(1, len(feature_cols)+1):
        for f in itertools.combinations(feature_cols, i):
            D = {'neighbors': n, 'features': f}
            combination_list.append(D) # append this combination to the list        
        

def player_pos_knn(D):
    #set feature cols and outcome.
    X1 = player_season_mean_stats[list(D['features'])]
    scaler.fit(X1) #scale. is there a way to move this outside the loop?
    X1_scaled = scaler.transform(X1)
    y1 = player_season_mean_stats['pos_num_simplified']
    
    knn = KNeighborsClassifier(n_neighbors=D['neighbors'])
    scores = cross_val_score(knn, X1_scaled, y1, cv=10, scoring='accuracy')
    acc = scores.mean()
    
#    pos_num_simplified_eval = pos_num_simplified_eval.append({'features':str(f), 
#                                                                     'n_neighbor': n, 
#                                                                     'accuracy': acc}, ignore_index=True)
    return {'features':str(D['features']), 'n_neighbor': D['neighbors'], 'accuracy': acc}

templist = [player_pos_knn(combination_list[0]), player_pos_knn(combination_list[-1])]

df = pd.DataFrame(pool.map(player_pos_knn, combination_list))

temp = pd.DataFrame(player_pos_knn(combination_list[-2]))
pos_num_simplified_eval = pos_num_simplified_eval.append(temp)

pos_num_simplified_eval 

    
pos_num_simplified_eval.shape

pos_num_simplified_eval.n_neighbor.unique #made it partially through neighbors = 1

pos_num_simplified_eval.sort_values('accuracy', ascending = False).to_csv('knn_metrics_player_pos.csv')

#testing to see if KNN works better when filtering the roster to players with more than n games
#use feature list and neighbors previously identified as the best
feature_n = ['assists', 'blocked', 'evenTimeOnIce_s', 'giveaways', 'goals', 'penaltyMinutes_s', 'plusMinus', 'powerPlayAssists', 'shortHandedTimeOnIce_s', 'shots', 'takeaways']
neighbor_n = 9

temp_df = player_season_mean_stats[player_season_mean_stats.games_played >= 10]
temp_df.head()
len(temp_df)

for i in range(1,51):
    temp_df = player_season_mean_stats[player_season_mean_stats.games_played >= i]

    #set feature cols and outcome.
    X1 = temp_df[feature_n]
    scaler.fit(X1) #scale. is there a way to move this outside the loop?
    X1_scaled = scaler.transform(X1)
    y1 = temp_df['pos_num_simplified']
    
    knn = KNeighborsClassifier(n_neighbors=9)
    scores = cross_val_score(knn, X1_scaled, y1, cv=10, scoring='accuracy')
    print "Minimum Games: %d, Accuracy: %f, N players: %d" % (i, scores.mean(), len(temp_df))
    
#increasing the minimum number of games leads to marginal increase in accuracy
#min games = 1 results in an accuracy of ~74% while 50 games results in an accuracy of ~77%


#see how the results change once we add in faceoff wins. Still using pos_num_simplified
feature_n = ['assists', 'blocked', 'evenTimeOnIce_s', 'giveaways', 'goals', 'penaltyMinutes_s', 'plusMinus', 'powerPlayAssists', 'shortHandedTimeOnIce_s', 'shots', 'takeaways', 'faceOffWins']

for i in range(1,51):
    temp_df = player_season_mean_stats[player_season_mean_stats.games_played >= i]

    #set feature cols and outcome.
    X1 = temp_df[feature_n]
    scaler.fit(X1) #scale. is there a way to move this outside the loop?
    X1_scaled = scaler.transform(X1)
    y1 = temp_df['pos_num_simplified']
    
    knn = KNeighborsClassifier(n_neighbors=9)
    scores = cross_val_score(knn, X1_scaled, y1, cv=10, scoring='accuracy')
    print "Minimum Games: %d, Accuracy: %f, N players: %d" % (i, scores.mean(), len(temp_df))

#accuracy increase of 10-15%

#see how the results change once we add in faceoff wins but look at pos_num again
feature_n = ['assists', 'blocked', 'evenTimeOnIce_s', 'giveaways', 'goals', 'penaltyMinutes_s', 'plusMinus', 'powerPlayAssists', 'shortHandedTimeOnIce_s', 'shots', 'takeaways', 'faceOffWins']

for i in range(1,51):
    temp_df = player_season_mean_stats[player_season_mean_stats.games_played >= i]

    #set feature cols and outcome.
    X1 = temp_df[feature_n]
    scaler.fit(X1) #scale. is there a way to move this outside the loop?
    X1_scaled = scaler.transform(X1)
    y1 = temp_df['pos_num']
    
    knn = KNeighborsClassifier(n_neighbors=9)
    scores = cross_val_score(knn, X1_scaled, y1, cv=10, scoring='accuracy')
    print "Minimum Games: %d, Accuracy: %f, N players: %d" % (i, scores.mean(), len(temp_df))
#drops back down to high 60s/low 70s. summary summary statistics are too obfuscated between wings and center to distinguish well
    
#get a better understanding of the distribution of games played
player_season_mean_stats.games_played.hist() #~15% of players play in 10 or fewer games. May want to consider excluding these from clusters
player_season_mean_stats.boxplot(column = 'games_played', by = 'position') #a lot of pairity across positions.