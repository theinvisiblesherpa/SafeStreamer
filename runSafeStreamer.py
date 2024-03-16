"""
Created on Mon Jan 15 10:48:57 2024

@author: JamHeinlein
"""

import streamlit as st
import pandas as pd
import os, sys
from sklearn.feature_extraction import DictVectorizer
from ModelTraining import trainNNModel,getFilmSuggestions
import pickle, joblib
import toml
import boto3

#with open("./AWS_CRED.toml","r") as f:
#    creds = toml.load(f)

s3 = boto3.resource(
    service_name='s3',
    region_name=os.environ["AWS_DEFAULT_REGION"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )

st.title('Safe Streamer')
st.markdown("Please rate the following movies, select your preferred genres, and select the content you wish to avoid. Click the button below and receive your recommendations!")

use_full = True
use_S3 = True

if use_full:
    setSize = "large"
else: 
    setSize = "small"

if not use_S3:
    # List of all potential Triggers
    with open('./Datasets/'+setSize+'/TrigList.pkl','rb') as trigPick:
        trigList = pickle.load(trigPick)

    # List of all potential Genres
    with open('./Datasets/'+setSize+'/GenreList.pkl','rb') as genrePick:
        genreList = pickle.load(genrePick)
    #    genreList.remove('IMAX')
    #    genreList.remove('(no genres listed)')


    if not os.path.isfile("./Datasets/"+setSize+"/filmWTrigs.pkl"):

        # List of all Films 
        with open('./Datasets/'+setSize+'/FilmList.pkl','rb') as filmPick:
            filmList = pickle.load(filmPick)
            filmList = filmList[["title","genres","tmdbId"]]

        # DF of Triggers in a given movie
        with open('./Datasets/'+setSize+'/TrigCond.pkl','rb') as trigDFPick:
            trigDF = pickle.load(trigDFPick)

        #Combine the TrigDF with the FilmList
        filmList = pd.merge(filmList, trigDF,how = 'left')
        filmList = filmList[["title","tmdbId","genres","Triggers"]]
        filmList = filmList.dropna()
        filmList = filmList.astype({'tmdbId':'int32'})

        #Clean up some NaNs for files without information
        filmList['Triggers'] = filmList['Triggers'].fillna("").apply(list)
        filmList['Triggers'] = filmList['Triggers'].apply(lambda x: ["None Known"] if len(x) == 0 else x)
        filmList['Triggers'] = filmList['Triggers'].apply(lambda x: ["None Known"] if x == [""] else x)

        with open('./Datasets/'+setSize+'/filmWTrigs.pkl','wb') as filmTrigs:
            pickle.dump(filmList,filmTrigs)

    with open('./Datasets/'+setSize+'/filmWTrigs.pkl','rb') as filmTrigs:
        filmList = pickle.load(filmTrigs)

    #Review Pickle
    with open('./Datasets/'+setSize+'/UserMovieDB.pkl','rb') as filmReviews:
        reviewDF = pickle.load(filmReviews)
        reviewDF = reviewDF[["title","userId","tmdbId","rating"]]

else:
    genreListS3 = s3.Bucket('safestreamerdata').Object('large/GenreList.pkl').get()
    genreList = pickle.load(genreListS3['Body'])

    trigListS3 = s3.Bucket('safestreamerdata').Object('large/TrigList.pkl').get()
    trigList = pickle.load(trigListS3['Body'])

    filmListS3 = s3.Bucket('safestreamerdata').Object('large/filmWTrigs.pkl').get()
    filmList = pickle.load(filmListS3['Body'])

    reviewDFS3 = s3.Bucket('safestreamerdata').Object('large/UserMovieDB.pkl').get()
    reviewDF = pickle.load(reviewDFS3['Body'])

tmdbIDs = pd.Series([862,8467,2493,680,1858,8966,597,9012,17473,603])
defaultMovies = {"Poster":['https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_.jpg',
               "https://m.media-amazon.com/images/I/91jhEwcQf2L._AC_UF894,1000_QL80_.jpg",
                "https://m.media-amazon.com/images/I/7116Aa2ZkRL._AC_UF894,1000_QL80_.jpg",
               "https://m.media-amazon.com/images/S/pv-target-images/dbb9aff6fc5fcd726e2c19c07f165d40aa7716d1dee8974aae8a0dad9128d392.jpg",
               "https://www.originalfilmart.com/cdn/shop/products/transformers_revenge_of_the_fallen_2009_original_film_art.webp?v=1678992764",
                "https://m.media-amazon.com/images/I/71cO0B8M+XL.jpg",
               "https://images-na.ssl-images-amazon.com/images/I/51G13d3EwBL._AC_UL600_SR600,600_.jpg",
               "https://m.media-amazon.com/images/I/41AlTxlPOXL._AC_UF894,1000_QL80_.jpg",
               "https://upload.wikimedia.org/wikipedia/en/thumb/e/e1/TheRoomMovie.jpg/220px-TheRoomMovie.jpg",
               "https://m.media-amazon.com/images/I/51DUmDryAvL._AC_UF894,1000_QL80_.jpg"
               ],

               "Title": ["Toy Story","Dumb and Dumber","The Princess Bride","Pulp Fiction", "Transformers","Twilight","Titanic","Jackass","The Room","The Matrix"],
    "Your Rating":[None,None]*5}
inputDB = pd.DataFrame(defaultMovies)

def filterMacro(selectFilterList, removeFilterList, dataframe):
   outDF = dataframe

   for thisFilter in selectFilterList:
         if not thisFilter:
           continue 
         outDF = selectFilter(thisFilter, outDF)
   for thisFilter in removeFilterList:
         if not thisFilter:
           continue 
         outDF = removeFilter(thisFilter, outDF)
   return outDF

# Filtering by rejecting selections
def removeFilter(filterOptions, dataframe):
    outDF = dataframe[dataframe["Triggers"].apply(lambda x: (not set(x).intersection(filterOptions ))).astype(bool)]  
    return outDF
    
# Filtering by prefering selections
def selectFilter(filterOptions, dataframe):
    outDF = dataframe[dataframe["genres"].apply(lambda x: set(x).intersection(filterOptions )).astype(bool)]  
    return outDF

# Return film suggestions based on user reviews
def printResults():
    filmSuggs = returnSuggestions()
    outDF = filterMacro(selectFilters,removeFilters,filmSuggs)
    st.dataframe(
        #filterMacro(selectFilters,removeFilters,filmList).head(20),
        outDF.head(10),
        #        filmList,
    hide_index=True
    )

# Formate the New User Reivews for NN Algo 
def formatUserReviews(inDF):

    # Add movie IDs  and give the user a default value (large)
    inDF["tmdbId"] = tmdbIDs
    inDF['userId'] = ["10000000"]*10

    inDF = inDF[["Title","Your Rating","userId","tmdbId"]]
    inDF = inDF.rename(columns={"Your Rating": "rating","Title":"title"}).fillna(0)
 
    return inDF

# Vectorize Reviews
def transformFeatures(inDF):

    dataMatrix = inDF.groupby("userId").apply(lambda items: {i.tmdbId: i.rating for i in items.itertuples()}) 
    features = DictVectorizer().fit_transform(dataMatrix)

    return features

# Create Model If Doesn't Exist
# Only train on the sample movies given to not penalize being similar to users who rated more ! 
if not os.path.isfile("./Datasets/"+setSize+"/NNModel.joblib"):
    newFeatures = transformFeatures(reviewDF[reviewDF['tmdbId'].isin([862,8467,2493,680,1858,8966,597,9012,17473,603])])
    trainedModel = trainNNModel(newFeatures)
    joblib.dump(trainedModel, "./Datasets/"+setSize+"/NNModel.joblib")
else:
    trainedModel = joblib.load("./Datasets/"+setSize+"/NNModel.joblib")

# Read in User Reviews
# Find Nearest Neighbors 
# Find Movies of NN
# Return results
def returnSuggestions():
   
    newUserReviews = formatUserReviews(userinfo)
    newUserFeatures = transformFeatures(newUserReviews)

    filmSuggestions = getFilmSuggestions(newUserFeatures, reviewDF, trainedModel)
    filmSuggestions = pd.merge(filmSuggestions,filmList, how='inner', on='tmdbId', suffixes=('','_copy'))[['title','genres','Triggers']]

    return filmSuggestions

with st.sidebar:
    trigSet = st.multiselect(
        'Content To Avoid:',
        trigList,
        [])
    genreSet = st.multiselect(
        'Genre:',
        genreList,
        [])

    selectFilters = [genreSet]
    removeFilters = [trigSet]
    allFilters = [selectFilters, removeFilters]

if __name__ == "__main__":
    
    with st.form("ReviewForm",border=False):
        userinfo = st.data_editor(
        inputDB,
        column_config={
            "Poster": st.column_config.ImageColumn(
                "Preview Image", help="Streamlit app preivew", width ="medium",
            ),
            "Your Rating": st.column_config.NumberColumn(
                min_value=1,
                max_value =5,
                step = 1.0,
                format = "%d ⭐"
            ),
            "Title": st.column_config.TextColumn(
                width="medium"
            )
        },
        disabled=["Poster","Title"],
        hide_index=True
        )
        submitted =  st.form_submit_button("Get My Results")
    if submitted:
        printResults()

