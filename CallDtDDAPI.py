#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:48:57 2024

@author: JamHeinlein
"""

import requests
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import time
import os

APIKey = os.environ["DtDD_API_KEY"]

headers = {'X-API-KEY': APIKey,
           'Accept': 'application/json'
           }

def getDtDDID(MovieName, tmdbID):
    params = {'q' : MovieName}
    thisResponse = requests.get('https://www.doesthedogdie.com/dddsearch' ,headers=headers, params=params).json()
    
    for movie in thisResponse['items']:    
        movieTMDBID = movie['tmdbId']
        
        if movieTMDBID == tmdbID:
            return movie['id']
    return np.nan
        
def voteMechanism(yesVotes, noVotes,debug=0):
    sumVotes = yesVotes + noVotes
    voteDiff = abs(yesVotes-noVotes)
    votePErr = math.sqrt(sumVotes)
    if voteDiff <= votePErr:
        if debug: print ("Unsure")
        return -1
    else:
        if yesVotes > noVotes:
            if debug: print ("Yes")
            return 1
        else:
            if debug: ("No")
            return 0

# Takes in DtDD ID
# Returns dictionary of triggers, their corresponding vote, and the raw yields (in case wanted later)
def getTrigInfo(movieID, tmdbID, debug=0):
    trigDict = defaultdict(lambda: [-1,0,0])

    if movieID is np.nan:
        return trigDict
    
    
    thisResponse = requests.get('https://www.doesthedogdie.com/media/'+str(movieID) ,headers=headers).json()
    movieName = thisResponse["item"]["name"]
    
    if debug: print (movieName)
    trigDict["tmdbId"] = tmdbID
    for trigger in thisResponse['topicItemStats']:

        trigName = trigger['topic']['name']
        yesVotes = trigger['yesSum']
        noVotes = trigger['noSum']
        decision = voteMechanism(yesVotes,noVotes)
        if debug:    
            print (trigName)
            print ("The votes are: Yes:", yesVotes, " No: ", noVotes, "Decision: ", decision)
            
        trigDict[trigName] = [decision, yesVotes, noVotes]
    return trigDict


# Takes in movie name and tmdbID nubmer 
# We may just need the movie name, but including the tmdbID ensures we got the right movie
def retrieveDtDDInfo(movieName, tmdbID):
    dtddID = getDtDDID(movieName, tmdbID)
    return getTrigInfo(dtddID, tmdbID)


def retrieveAllDtDDInfo(movies):
    trigDict = {}
    for entry in movies:
        print (entry[0], entry[1])
        print("Fetching ", entry[0], "........")
        time.sleep(0.1)
        thisTriggers = retrieveDtDDInfo(entry[0], entry[1])
        
        trigDict[entry[0]] = thisTriggers
        
    return pd.DataFrame.from_dict(trigDict, orient='index')
