import pandas as pd
import simpsons as s
from sklearn.externals import joblib

def load_data():
    '''
    INPUT: nothing
    OUTPUT: dataframe from `_add_predicts()`` fuction below

    Loads in the two models, the episode DataFrame, the scripts, and returns a lear Dataframe with all the features in place with the predictions on Adaboost and Gradient Boosted.
    '''
    # load both models from stack in
    abm = joblib.load("pickled_models/adaboost_model.pkl")
    gbm = joblib.load("pickled_models/stacked_model.pkl")
    # read in episode data to Pandas DataFrame
    e_filename = 'data/simpsons_episodes.csv'
    # get titles and id for each episode
    titles = pd.read_csv(e_filename)[['id', 'title']]
    # read in raw episode data, return clean episode pandas dataframe
    episode_df = s.return_clean_script_df("data/simpsons_script_lines.csv", s.clean_episode_data(e_filename))
    #fill NaNs
    df = s._fill_nans(episode_df)
    return _add_predicts(df, abm, gbm, titles)

def _add_predicts(df, abm, gbm, titles):
    '''
    INPUT: DF with no NaNs, both models for stacked model in form `model 1, model 2`, a Pandas DataFrame with the title and id of episodes for the TV show. (all passed from load_data() function above.)
    OUTPUT: The combination DF of all episodes with all features and the predictions for both the first model and the second.
    '''
    # make DF with columns to be returned in suggestion
    df2 = df[['id', 'views', 'imdb_rating', 'image_url', 'video_url', 'season']]
    # drop all the columns that aren't helpful
    df.drop(labels=[x for x in df.columns.tolist() if x not in ['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']], axis=1, inplace=True)
    # make sure I have the correct columns
    df = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']]
    # predict using the model stack of `abm` / `gbm`
    df['abm'] = abm.predict(df)
    df['predicted'] = gbm.predict(df)
    # remove the initial prediction (from `abm`)
    df.drop('abm', axis=1, inplace=True)
    #combine the dataframes using a left join
    df = df.merge(df2, how='left', left_on = 'id', right_on = 'id')
    df = df.merge(titles, how='inner', left_on = 'id', right_on = 'id')
    return df

if __name__ == '__main__':
    load_data()
