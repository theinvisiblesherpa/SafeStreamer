# SafeStreamer
A Streamlit webapp for generating movie recommendations while avoiding triggering topics

This application is intended for people who have difficulty selecting a movie to watch amongst the endless sea of content available across numerous streaming services. Unlike other recommendation engines, Safe Streamer allows for the filtering of movies containing unwanted/triggering content (eg pet death, SA, etc). The user is asked to rate a sample of 10 preselected movies along with the triggering content they wish to avoid. SafeStreamer then returns a list of movies which similar users enjoy, organized by highest expected rating from the user. This aids the user by both speeding up the process of selecting a movie while also helping them avoid unwanted content. 

## Dependencies
SafeStreamer requires the following packages to function:

'streamlit==1.30.0'

'numpy==1.23.5'

'pandas==2.2.1'

'Requests==2.31.0'

'scikit_learn==1.3.0'

## Running Instructions
SafeStreamer can be run via the following command:
'streamlit run runSafeStreamer.py` 

For lower memory usage, 'runSafeStreamer.py' can be edited in Line 17 by changing use_full to False. This lowers the size of the dataset utilized to generate suggestions down to 610 unique users.

## Under The Hood

Safe Streamer takes the [MovieLens review database](https://grouplens.org/datasets/movielens/), which contains approximately 33,000,000 movie reviews from 330,975 unique users. Content warnings are sourced from user-voted website [DoesTheDogDie.com](DoesTheDogDie.com), which is accessed via their [API](https://www.doesthedogdie.com/api) in CallDtDDAPI.py. To get a sense of new user taste, a set of 10 movies was selected to best differentiate a users interest. These 10 movies were selected by performing exploratory data analysis in FilmEDA.ipynb to find movies with high viewership as well as high review variance. From here, a nearest neighbors algorithm is trained on the original set of MovieLens data, with hyperparameters tuned manually (given this is unsupervised learning). Once the user has reviewed the initial set of films, the input data is formatted for compatibility with the NN algo and the closest 20 neighbors via a cosine similarity metric are pulled. The film reviews of the 20 neighbors are collected together and a baysian mean is performed for each movie independently, with the global average review score chosen as the baysian prior. The returned set of films is then filtered based on genre preference and trigger avoidance and displayed to the user. This application is deployed via Streamlit using simple interactables like st.data_editor, st.button, st.sidebar. To avoid needing to collect the suggestions after every user review (Streamlit updates every user click), the input dataset is implemented as an st.form to delay this unnecessary processing.
