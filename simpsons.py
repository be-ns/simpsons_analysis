import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as g_br
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import Imputer
from fancyimpute import KNN
from math import sqrt

def clean_episode_data(filename):
    df = pd.read_csv(filename)
    # drop columns that provide nothing useful for quantitative analysis
    df.drop('image_url', axis=1, inplace = True)
    df.drop('video_url', axis=1, inplace = True)
    df.drop('imdb_votes', axis=1, inplace=True)
    df.drop('production_code', axis=1, inplace=True)

    # reduce views to a proportion, equal to number of total views over number of viewers in the USA (this number is in millions, so it will always allow a positive value here)
    df['views'] = df['views'] / df['us_viewers_in_millions']
    df.drop('us_viewers_in_millions', axis=1, inplace=True)

    # turn the title into a metric for lenght of episode title
    df['title'] = df['title'].apply(lambda x: len(x))

    # currently dropping start date, but this could change if insightful - maybe for political cycles?
    df['original_air_date'] = pd.to_datetime(df['original_air_date'])
    ## df.drop('original_air_date', axis=1, inplace=True)

    # ideally NaN Imputing could be done using a more accurate measure such as KNN
    df.fillna(method = 'bfill', inplace=True)
    return df

def add_location_data(fname, edf):
    ldf = pd.read_csv(fname)
    # keep only the normalized name so capitalization isn't a problem
    ldf.drop('name', axis=1, inplace=True)
    return None

def return_clean_script_df(fname):
    # import errors for imconsistent use of quotations removed by `error_bad_lines` call
    script_df = pd.read_csv(fname, error_bad_lines = False)
    # script_df.dropna(axis=0, inplace=True)
    # keep only the normalized lines so capitalization isn't a problem
    script_df.drop('spoken_words', axis=1, inplace=True)
    # remove the uncleaned, raw script lines
    script_df.drop('raw_text', axis=1, inplace=True)

    # print(script_df.groupby)
    return script_df


if __name__ == '__main__':
    e_filename = '/Users/benjamin/Desktop/DSI/capstone/the-simpsons-by-the-data/simpsons_episodes.csv'
    episode_df = clean_episode_data(e_filename)
    # read in raw episode data, return clean episode pandas dataframe
    print(episode_df.head())
    # l_filename = '/Users/benjamin/Desktop/DSI/capstone/the-simpsons-by-the-data/simpsons_locations.csv'
    # updated_episode_df = add_location_data(l_filename, episode_df)
    # # print(locations_df.head())
    # script_df = return_clean_script_df('/Users/benjamin/Desktop/DSI/capstone/the-simpsons-by-the-data/simpsons_script_lines.csv')
    # print(script_df.head())
