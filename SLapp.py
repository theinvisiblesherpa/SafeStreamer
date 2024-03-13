import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction import DictVectorizer
from ModelTraining import trainNNModel,getFilmSuggestions

st.title('Safe Streamer')
st.markdown("Please rate the following movies, select your preferred genres, and select the content you wish to avoid. Click the button below and receive your recommendations!")
import pickle

use_full = False

if use_full:
    setSize = "large"
else: 
    setSize = "small"

# List of all potential Triggers
with open('./Datasets/'+setSize+'/TrigList.pkl','rb') as trigPick:
    trigList = pickle.load(trigPick)

# List of all potential Genres
with open('./Datasets/'+setSize+'/GenreList.pkl','rb') as genrePick:
    genreList = pickle.load(genrePick)
    genreList.remove('IMAX')
    genreList.remove('(no genres listed)')

if not os.path.isfile("./Datasets/"+setSize+"/filmWTrigs.pkl"):

    # List of all Films 
    with open('./Datasets/'+setSize+'/FilmList.pkl','rb') as filmPick:
        filmList = pickle.load(filmPick)
        filmList = filmList[["title","genres","tmdbId"]]

    # DF of Triggers in a given movie
    with open('./Datasets/'+setSize+'/TrigCond.pkl','rb') as trigDFPick:
        trigDF = pickle.load(trigDFPick)

    # Combine the TrigDF with the FilmList
    filmList = pd.merge(filmList, trigDF,how = 'left')
    filmList = filmList[["title","tmdbId","genres","Triggers"]]
    filmList = filmList.dropna()
    filmList = filmList.astype({'tmdbId':'int32'})

    # Clean up some NaNs for files without information
    filmList['Triggers'] = filmList['Triggers'].fillna("").apply(list)
    filmList['Triggers'] = filmList['Triggers'].apply(lambda x: ["None Known"] if len(x) == 0 else x)
    filmList['Triggers'] = filmList['Triggers'].apply(lambda x: ["None Known"] if x == [""] else x)

    with open('./Datasets/'+setSize+'/filmWTrigs.pkl','wb') as filmTrigs:
        pickle.dump(filmList,filmTrigs)

with open('./Datasets/'+setSize+'/filmWTrigs.pkl','rb') as filmTrigs:
    filmList = pickle.load(filmTrigs)

# Review Pickle
with open('./Datasets/'+setSize+'/UserMovieDB.pkl','rb') as filmReviews:
    reviewDF = pickle.load(filmReviews)
    reviewDF = reviewDF[["title","userId","tmdbId","rating"]]

tmdbIDs = pd.Series([862,8467,2493,680,1858,8966,597,1795,17473,603])
d = {"Poster":['https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_.jpg',
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
thisData = pd.DataFrame(d)

userinfo = st.data_editor(
    thisData,
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
    filmSuggs = returnSuggestions(userinfo,reviewDF)
    outDF = filterMacro(selectFilters,removeFilters,filmSuggs)
    st.dataframe(
        #filterMacro(selectFilters,removeFilters,filmList).head(20),
        outDF.head(10),
        #        filmList,
    hide_index=True
    )

# Add New User to Review DB 
def addUserReviews(inDF,reviewDF):
    # Add movie IDs  and give the user a default value (large)
    inDF["tmdbId"] = tmdbIDs
    inDF['userId'] = ["10000000"]*10
    inDF.index=["10000000"]*10
    inDF = inDF[["Title","Your Rating","userId","tmdbId"]]

    inDF = inDF.rename(columns={"Your Rating": "rating","Title":"title"})
    
    updatedDF = pd.concat([inDF,reviewDF], ignore_index=True)
 
    return updatedDF

# Vectorize Reviews
def transformFeatures(inDF):

    dataMatrix = inDF.groupby("userId").apply(lambda items: {i.tmdbId: i.rating for i in items.itertuples()}) 
    features = DictVectorizer().fit_transform(dataMatrix)

    return features, dataMatrix

# Read in User Reviews
# Add user reviews to existing DB
# Train KNN Algo
# Return results
def returnSuggestions(inDF, revDF):
   
    newUserReviews = addUserReviews(inDF,revDF)
    newUserReviews = newUserReviews.fillna(0)
    
    # Only train on the sample movies given to not penalize being similar to users who rated more ! 
    trainingReviews = newUserReviews[newUserReviews['tmdbId'].isin([862,8467,2493,680,1858,8966,597,1795,17473,603])]
    #newFeatures, newDataMatrix = transformFeatures(newUserReviews)
    newFeatures, newDataMatrix = transformFeatures(trainingReviews)

    newUserID = newFeatures.shape[0]-1 # New user is added to the end of the DF 
    trainedModel = trainNNModel(newFeatures)

    filmSuggestions = getFilmSuggestions(newUserID, newFeatures, newUserReviews, reviewDF, trainedModel)
#    filmSuggestions = getFilmSuggestions(newUserID, newFeatures, newDataMatrix, reviewDF, trainedModel)
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

st.button("Get My Results!", on_click=printResults)
