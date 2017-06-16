# needs to make all edits to data
# get predicted values and real values for IMDB rating
# needs to have both image and video url

import pandas as pd
import simpsons as s
from sklearn.externals import joblib

def load_data():
    abm = joblib.load("pickled_models/adaboost_model.pkl")
    gbm = joblib.load("pickled_models/stacked_model.pkl")
    # read in episode data to Pandas DataFrame
    e_filename = 'data/simpsons_episodes.csv'
    titles = pd.read_csv(e_filename)[['id', 'title']]
    # read in raw episode data, return clean episode pandas dataframe
    episode_df = s.return_clean_script_df("data/simpsons_script_lines.csv", s.clean_episode_data(e_filename))
    df = s._fill_nans(episode_df)
    return _add_predicts(df, abm, gbm, titles)

def _add_predicts(df, abm, gbm, titles):
    df2 = df[['id', 'views', 'imdb_rating', 'image_url', 'video_url', 'season']]
    df.drop(labels=[x for x in df.columns.tolist() if x not in ['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']], axis=1, inplace=True)
    df = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']]
    df['abm'] = abm.predict(df)
    df['predicted'] = gbm.predict(df)
    df.drop('abm', axis=1, inplace=True)
    df = df.merge(df2, how='left', left_on = 'id', right_on = 'id')
    df = df.merge(titles, how='inner', left_on = 'id', right_on = 'id')
    return df

if __name__ == '__main__':
    load_data()
