#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:03:36 2024

@author: JamesHeinlein
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

use_full = False
if use_full:
    setSize = "large"
else: 
    setSize = "small"

def bayes_sum(N, mu):
    return lambda x: (x.sum() + mu*N) / (x.count() + N)

movies = pd.read_csv("./Datasets/"+setSize+"/movies.csv")
links = pd.read_csv("./Datasets/"+setSize+"/links.csv",usecols=['movieId','tmdbId'])
ratings = pd.read_csv("./Datasets/"+setSize+"/ratings.csv",usecols=['userId','movieId','rating'])
trigDF = pd.read_pickle("./Datasets/"+setSize+"/TrigInfo.pkl")
#tags = pd.read_csv("./Datasets/"+setSize+"/tags.csv")

# Make the Used DFs and remove the rest
moviesWLinks = pd.merge(movies, links)
userReviews = pd.merge(ratings, moviesWLinks, how="left")
del movies, links

with open("./Datasets/"+setSize+"/UserMovieDB.pkl", 'wb') as UserMoviePick:    
    pickle.dump(userReviews, UserMoviePick)


# Train nearest neighbors model
def trainNNModel(inData):
    userModel= NearestNeighbors(n_neighbors=25,
                                   metric='cosine',
                                   algorithm='brute',
                                   n_jobs=-1)
    userModel.fit(inData)
    return userModel

def getFilmSuggestions(userID, features, dataMatrix, revDF, model):
    # Get nearest neighbors and collect all the reviews associated to neighbors
    dists, indices = model.kneighbors(features[dataMatrix.index.get_loc(userID), :])
    neighbors = [dataMatrix.index[i] for i in indices[0]][1:]
    ratings_grp = revDF[revDF['userId'].isin(neighbors)].groupby('title')[['tmdbId','rating']]
    
    suggestions = ratings_grp.aggregate({'rating':bayes_sum(1, 3.69),'tmdbId':'mean'}).sort_values(by='rating',ascending=False)
    
    # Optional other metric
    #suggestions = ratings_grp.aggregate({'rating':'count','tmdbId':'mean'}).sort_values(by='rating',ascending=False)

    suggestions = suggestions.dropna()
        
    # Dont recommend movies from the base list
    suggestions = suggestions.astype({'tmdbId':'int32'})
    suggestions = suggestions[~suggestions['tmdbId'].isin([862,8467,2493,680,1858,8966,597,1795,17473,603])]
    return suggestions
