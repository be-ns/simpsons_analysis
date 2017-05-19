import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as g_br
from sklearn.ensemble import RandomForestRegressor as r_fr
from sklearn.ensemble import AdaBoostRegressor as a_br
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import Imputer
from math import sqrt

def clean_episode_data(filename):
    e_filename = '/Users/benjamin/Desktop/DSI/capstone/simpsons_analysis/data/simpsons_episodes.csv'

    df = pd.read_csv(e_filename).sort_values(by='id')
    # initial dataframe cleaning
    # drop columns that provide nothing useful for quantitative analysis
    df.drop('image_url', axis=1, inplace = True)
    df.drop('video_url', axis=1, inplace = True)
    df.drop('imdb_votes', axis=1, inplace=True)
    df.drop('production_code', axis=1, inplace=True)
    # reduce views to a proportion, equal to number of total views over number of viewers in the USA (this number is in millions, so it will always allow a positive value here)
    df['views'] = df['us_viewers_in_millions']
    df.drop('us_viewers_in_millions', axis=1, inplace=True)
    # turn the title into a metric for lenght of episode title
    df['title_len'] = df['title'].apply(lambda x: len(x))
    df.drop('title', axis=1, inplace=True)
    # currently dropping start date, but this could change if insightful - maybe for political cycles?
    df['original_air_date'] = pd.to_datetime(df['original_air_date'])
    # ideally NaN Imputing could be done using a more accurate measure such as KNN
    df.fillna(method = 'bfill', inplace=True)
    df['election_year'] = df['original_air_date'].apply(lambda x: 1 if x.year % 4 == 0 else 0)
    return df

def add_location_data(fname, edf):
    ldf = pd.read_csv(fname)
    # keep only the normalized name so capitalization isn't a problem
    ldf.drop('name', axis=1, inplace=True)
    return None

def return_clean_script_df(fname, df):
    # import errors for imconsistent use of quotations removed by `error_bad_lines` call
    script_df = pd.read_csv(fname, error_bad_lines = False)
    # print(script_df.columns.tolist())
    script_df.dropna(axis=0, inplace=True)
    # keep only the normalized lines so capitalization isn't a problem
    script_df.drop('spoken_words', axis=1, inplace=True)
    # remove the uncleaned, raw script lines
    script_df.drop('raw_text', axis=1, inplace=True)
    # get totals for the number of episodes of lines in each episode
    line_series = script_df.groupby('episode_id')['speaking_line'].count()
    line_length_series = script_df.groupby('episode_id')['word_count'].mean()
    # make sure it has the same column name for pandas merge
    # print(new_df.head())
    # new_df.drop('episode_id', axis=1, inplace=True)
    # unfortunately we are missing
    df = df.merge(pd.DataFrame(line_series), how='left', left_on = 'id', right_index=True)
    df = df.merge(pd.DataFrame(line_length_series), how='left', left_on = 'id', right_index=True)
    return df

def try_stacking_models(df):
    # train test split model testing model stacking (adding result of one or more models as a datapoint for the next model)
    # fill all NaNs
    df.views.fillna(df.views.mean(), inplace=True)
    df.imdb_rating.fillna(df.imdb_rating.mean(), inplace=True)
    df.speaking_line.fillna(df.speaking_line.mean(),inplace=True)
    df.word_count.fillna(df.word_count.mean(),inplace=True)
    # print(df.head())
    # get X and y
    y = df['imdb_rating']
    X = df[['season', 'number_in_series', 'views',  'title_len', 'election_year', 'speaking_line', 'word_count']]
    # build stacked model using Gradient Boost, AdaBoost, and Random Forest
    gbr, rfr, abr = g_br(n_estimators=1000, max_depth = 5, max_features=5), r_fr(n_estimators=200, max_features = 5, n_jobs=-1, verbose=True), a_br(n_estimators=1000, learning_rate=0.7)
    # go through and build model, then recreate dataframe - this should be in a pipeline though - I'll get to that
    gbr.fit(X, y)
    X['gbr_score'] = gbr.predict(X)
    abr.fit(X, y)
    X['abr_score'] = abr.predict(X)
    X_tr, X_te, y_tr, y_te = tts(X, y)
    rfr.fit(X_tr, y_tr)
    pred = rfr.predict(X_te)
    # return test vals
    return str('final score (rmse) = {0}'.format(sqrt(mse(y_te, pred))))


if __name__ == '__main__':
    # read in episode data to Pandas DataFrame
    e_filename = '/Users/benjamin/Desktop/DSI/capstone/simpsons_analysis/data/simpsons_episodes.csv'
    episode_df = clean_episode_data(e_filename)
    # read in raw episode data, return clean episode pandas dataframe
    loc_filename = '/Users/benjamin/Desktop/DSI/capstone/simpsons_analysis/data/simpsons_locations.csv'
    # updated_episode_df = add_location_data(loc_filename, episode_df)
    episode_df = return_clean_script_df("/Users/benjamin/Desktop/DSI/capstone/simpsons_analysis/data/simpsons_script_lines.csv", episode_df)
    # get rmse for stacked model
    print(try_stacking_models(episode_df))
