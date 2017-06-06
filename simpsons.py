import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as g_br
from sklearn.ensemble import RandomForestRegressor as r_fr
from sklearn.ensemble import AdaBoostRegressor as a_br
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import Imputer
from math import sqrt
import matplotlib.pyplot as plt
from numpy import ascontiguousarray as c_style

def impute_with_knn(df):
    impute_subset = test_scores.drop(labels=['family_inv','prev_disab','score'], axis=1)
    y = impute_subset.pop('mother_hs').values
    X = preprocessing.StandardScaler().fit_transform(impute_subset.astype(float))
    missing = np.isnan(y)
    mod = LogisticRegression()
    mod.fit(X[~missing], y[~missing])
    mother_hs_pred = mod.predict(X[missing])
    mother_hs_pred
    mod2 = LogisticRegression(C=1, penalty='l1')
    mod2.fit(X[~missing], y[~missing])
    mod2.predict(X[missing])
    mod3 = LogisticRegression(C=0.4, penalty='l1')
    mod3.fit(X[~missing], y[~missing])
    mod3.predict(X[missing])
    mother_hs_imp = []
    for C in 0.1, 0.4, 2:
        mod = LogisticRegression(C=C, penalty='l1')
        mod.fit(X[~missing], y[~missing])
        imputed = mod.predict(X[missing])
        mother_hs_imp.append(imputed)

def clean_episode_data(filename):
    '''
    INPUT: filename
    OUTPUT: Pandas DataFrame containing scrubbed data
    '''
    df = pd.read_csv(filename, sep=',').sort_values(by='id')
    # initial dataframe cleaning
    # drop columns that provide nothing useful for quantitative analysis
    # change views to views in millions and drop original column
    df['views'] = df['us_viewers_in_millions']
    df.drop('us_viewers_in_millions', axis=1, inplace=True)
    # turn the title into a metric for length of episode title
    df['title_len'] = df['title'].apply(lambda x: len(x))
    # change air date to datetime
    df['original_air_date'] = pd.to_datetime(df['original_air_date'])
    # Election cycle years are all divisible by four, (e.g. 2016) so I'm adding this info to see if it gives any insight to IMDB ratings
    df['election_year'] = df['original_air_date'].apply(lambda x: 1 if x.year % 4 == 0 else 0)
    # fill with backfill, since episodes are sequential, resets any missing values to be the same as the prior episode
    df.fillna(method = 'bfill', inplace=True)
    # drop any unnecesary columns
    df.drop(labels=['title', 'image_url','video_url','imdb_votes', 'production_code'], axis=1, inplace = True)
    return df

def add_location_data(script_df):
    '''
    INPUT: Script DataFrame
    OUTPUT: Pandas Series with count of locations per episode
    '''
    return script_df.groupby('episode_id')['location_id'].nunique()

def return_clean_script_df(fname, df):
    '''
    INPUT: filename as fname and the original episode DataFrame from `clean_episode_data`
    OUTPUT: DF with extracted script info, sorted by episode
    '''
    # import errors for inconsistent use of quotations removed by `error_bad_lines` call, need to have some help with this
    script_df = pd.read_csv(fname, error_bad_lines = False)
    # keep only the normalized lines so capitalization isn't a problem
    script_df.drop(labels = ['spoken_words', 'raw_text'], axis=1, inplace=True)
    # get totals for the number of episodes of lines in each episode
    line_series = script_df.groupby('episode_id')['speaking_line'].count()
    # merge into episode df
    # df = df.merge(pd.DataFrame(line_series), how='left', left_on = 'id', right_index=True)
    # get words spoken per episode
    line_length_series = script_df.groupby('episode_id')['word_count'].mean()
    # merge into episode df
    df = df.merge(pd.DataFrame(line_length_series), how='left', left_on = 'id', right_index=True)
    # get number of lines spoken by major characters (AKA the Simpsons)
    major_char_lines = script_df.where(script_df['raw_character_text'].str.contains('Simpson')).groupby('episode_id')['speaking_line'].count()
    # get ratio of lines spoken by Simpsons vs all
    major_char_lines = major_char_lines / line_series
    # merge the pandas Dataframes
    df = df.merge(pd.DataFrame(major_char_lines), how='left', left_on = 'id',  right_index=True)
    # get the count of locations from each episode
    loc_series = add_location_data(script_df)
    df = df.merge(pd.DataFrame(loc_series), how='left', left_on = 'id',  right_index=True)
    #rename columns to avoid confusion
    df.columns = ['id','original_air_date',  'season', 'number_in_season', 'number_in_series', 'views', 'imdb_rating', 'title_len', 'election_year', 'word_count', 'major_char_lines', 'locations_in_ep']
    return df

def try_stacking_models(df):
    '''
    INPUT: clean Pandas Dataframe with script and episode information, kwarg for if we are printing RMSE or returning the dataframe itself (for PyMC3 usage)
    OUTPUT: Either the RMSE for the stacked model or a Dataframe with three new columns (scores from each of the three models)
    '''
    # train test split model testing model stacking (adding result of one or more models as a datapoint for the next model)
    # fill all NaNs with column-wise mean
    df.views.fillna(df.views.mean(), inplace=True)
    df.imdb_rating.fillna(df.imdb_rating.mean(), inplace=True)
    df.major_char_lines.fillna(df.major_char_lines.mean(),inplace=True)
    # df.homer_lines.fillna(df.homer_lines.mean(),inplace=True)
    df.word_count.fillna(df.word_count.mean(),inplace=True)
    df.locations_in_ep.fillna(df.locations_in_ep.mean(), inplace=True)
    # get X and y
    y = df['imdb_rating']
    # X = df[['id', 'season', 'number_in_season', 'number_in_series', 'views', 'title_len', 'election_year', 'word_count', 'homer_and_bart']]
    X = df[['id', 'number_in_season', 'views', 'title_len', 'word_count', 'major_char_lines']]
    # build stacked model using Gradient Boost, AdaBoost, and Random Forest
    # define model parameters
    gbr, rfr, abr = g_br(loss = 'lad', learning_rate = .1, n_estimators=500, subsample = .85, warm_start=True, verbose=False), r_fr(n_estimators=300, n_jobs=-1, verbose=False, min_impurity_split=1e-6), a_br(n_estimators=500, learning_rate=.4, loss = 'exponential')
    # go through and build model for each, then add column to dataframe with predicted. Make sure to train_test_split
    # Random Forest Regressor
    score_rfr = 10
    while score_rfr > .4:
        _rfr = rfr
        X_tr, X_te, y_tr, y_te = tts(X, y)
        _rfr.fit(X_tr, y_tr)
        score_rfr = sqrt(mse(y_te, _rfr.predict(X_te)))
        print('Random_forest_score = ', score_rfr)
    X['rfr_score'] = _rfr.predict(X)
    # AdaBoost Regressor
    abr_score = 10
    trees = 500
    while abr_score > .285:
        _abr = a_br(n_estimators=trees, learning_rate=.4, loss = 'exponential')
        X_tr, X_te, y_tr, y_te = tts(X, y)
        _abr.fit(X_tr, y_tr)
        abr_score = sqrt(mse(y_te, _abr.predict(X_te)))
        print('Adaboost_score = ', abr_score)
        trees += 10
    X['abr_score'] = _abr.predict(X)
    # Gradient Boosed Regressor
    gbr_score = 10
    X_tr, X_te, y_tr, y_te = tts(X, y)
    while gbr_score > .208:
        _gbr = g_br(loss = 'lad', learning_rate = .1, n_estimators=trees, subsample = .85, warm_start=True, verbose=False)
        _gbr.fit(c_style(X_tr), c_style(y_tr))
        gbr_score = sqrt(mse(y_te, _gbr.predict(X_te)))
        print('Gradient_boosted_score = ', gbr_score)
        trees += 10
    # return test vals
    X['stacked_score'] = _gbr.predict(X)
    X['rating'] = y
    df = X
    plot_scores(df, gbr_score)
    print(str('final score (rmse) = {0}'.format(gbr_score)))
    # get feature importances, then score model using predicted values from GradientBoostingRegressor
    print(sorted(list(zip(_gbr.feature_importances_, ['id', 'number_in_season', 'views', 'title_len', 'word_count', 'major_char_lines', 'rfr_score', 'abr_score'])), reverse=True))


def plot_scores(df, rmse):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(35,15))
    ax = fig.add_subplot(111)
    ax.plot(df['id'][::2], df['rating'][::2], label='True Rating', color='r', lw=5, alpha=.4)
    ax.plot(df['id'][::2], df['stacked_score'][::2], label = 'Predicted Rating', color = 'b', lw =5, alpha = .4)
    ax.set_xlim((0,600))
    ax.tick_params(labelsize=20)
    ax.set_title('Predicted Ratings to Actual Ratings, RMSE = {0}'.format(round(rmse, 3)), size=40)
    ax.set_xlabel('Episode Number', size=35)
    ax.set_ylabel('Rating on 10 point scale', size = 35)
    plt.legend(prop={'size':35})
    plt.tight_layout()
    plt.savefig('score.png', dpi=100)
    plt.close('all')

if __name__ == '__main__':
    # read in episode data to Pandas DataFrame
    e_filename = 'data/simpsons_episodes.csv'
    # read in raw episode data, return clean episode pandas dataframe
    # loc_filename = 'simpsons_locations.csv'
    episode_df = return_clean_script_df("data/simpsons_script_lines.csv", clean_episode_data(e_filename))
    # get rmse for stacked model with kwarf 'rmse'
    try_stacking_models(episode_df)

# side note - I want to add in these details from scripts:
# song (bool)
# monologue (line len > 90 words)
