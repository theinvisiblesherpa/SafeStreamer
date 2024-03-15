#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:46:33 2023

@author: JamHeinlein
"""

import pandas as pd
import CallDtDDAPI as dtdd
import os,sys
import pickle

use_full = False

if use_full:
    setSize = "large"
else: 
    setSize = "small"

movies = pd.read_csv("./Datasets/"+setSize+"/movies.csv")
links = pd.read_csv("./Datasets/"+setSize+"/links.csv",usecols=['movieId','tmdbId'])
ratings = pd.read_csv("./Datasets/"+setSize+"/ratings.csv",usecols=['userId','movieId','rating'])
#trigDF = pd.read_pickle("./Datasets/"+setSize+"/TrigInfo.pkl")

moviesWLinks = pd.merge(movies, links)
userReviews = pd.merge(ratings, moviesWLinks, how="left")
del movies, links

with open("./Datasets/"+setSize+"/UserMovieDB.pkl", 'wb') as userMovieDB:    
    pickle.dump(userReviews, userMovieDB)

def saveDtDDInfo(movieDF):
    moviesWtmdbID = movieDF[["title","tmdbId"]].dropna()

    moviesWtmdbID["title"] = moviesWtmdbID["title"].apply(lambda x: x[:-7])
    moviesWtmdbID['tmdbId'] = moviesWtmdbID['tmdbId'].apply(lambda x: int(x))
    
    testSet = (moviesWtmdbID[:12])
    
    moviesWtmdbID = list(moviesWtmdbID.itertuples(index=False))
    testSet = list(testSet.itertuples(index=False))

    triggerDF = dtdd.retrieveAllDtDDInfo(moviesWtmdbID)
    
    return triggerDF


if not os.path.isfile("./Datasets/"+setSize+"/TrigInfo.pkl"):
    trigDF = saveDtDDInfo(moviesWLinks)
    trigDF.to_pickle("./Datasets/"+setSize+"/TrigInfo.pkl")
else:
    trigDF = pd.read_pickle("./Datasets/"+setSize+"/TrigInfo.pkl")

# Interpret Results From DtDD into adding a category column #
triggers = trigDF.columns[1:]
for trigger in triggers:
    trigDF[trigger] = trigDF[trigger].apply(lambda x : x[0])
    nPoints = trigDF[trigger].sum(axis=0)+trigDF[trigger].shape[0]
trigDF['Triggers'] = trigDF[triggers].apply(lambda x: list(','.join(x.index[x == 1]).split(",")), axis=1)
#trigDF['Triggers'] = trigDF[triggers].apply(lambda x: np.nan if len(x) == 0 else x)



# Get List of Genres #
moviesWLinks['genres'] = moviesWLinks['genres'].apply(lambda x: x.split("|"))
genreList = set(moviesWLinks['genres'].sum())



# Save Trigger and Genre Info Out
with open("./Datasets/"+setSize+"/TrigCond.pkl", 'wb') as trigDFPick:    
    pickle.dump(trigDF[["tmdbId","Triggers"]], trigDFPick)
with open("./Datasets/"+setSize+"/TrigList.pkl", 'wb') as trigPick:    
    pickle.dump(triggers, trigPick)
with open("./Datasets/"+setSize+"/GenreList.pkl", 'wb') as GenrePick:    
    pickle.dump(genreList, GenrePick)
with open("./Datasets/"+setSize+"/FilmList.pkl", 'wb') as FilmPick:
    pickle.dump(moviesWLinks,FilmPick)
