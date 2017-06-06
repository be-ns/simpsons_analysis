import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as g_br
from sklearn.ensemble import RandomForestRegressor as r_fr
from sklearn.ensemble import AdaBoostRegressor as a_br
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs
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
    monologues = script_df.groupby('episode_id')['word_count'].max()
    # merge into episode df
    df = df.merge(pd.DataFrame(line_length_series), how='left', left_on = 'id', right_index=True)
    df = df.merge(pd.DataFrame(monologues), how='left', left_on = 'id', right_index=True)
    # get number of lines spoken by major characters (AKA the Simpsons)
    major_char_lines = script_df.where(script_df['raw_character_text'].str.contains('Simpson')).groupby('episode_id')['speaking_line'].count()
    # get ratio of lines spoken by Simpsons vs all
    major_char_lines = major_char_lines / line_series
    # merge the pandas Dataframes
    df = df.merge(pd.DataFrame(major_char_lines), how='left', left_on = 'id',  right_index=True)
    # get the count of locations from each episode
    loc_series = add_location_data(script_df)
    df = df.merge(pd.DataFrame(loc_series), how='left', left_on = 'id',  right_index=True)
    print(df.columns.tolist())
    #rename columns to avoid confusion
    df.columns = ['id', 'original_air_date', 'season', 'number_in_season', 'number_in_series', 'views', 'imdb_rating', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']
    return df

def stack_models(df):
    '''
    INPUT: clean Pandas Dataframe with script and episode information, kwarg for if we are printing RMSE or returning the dataframe itself (for PyMC3 usage)
    OUTPUT: Either the RMSE for the stacked model or a Dataframe with three new columns (scores from each of the three models)
    '''
    # initialize tree count
    trees = 5000
    # train test split model testing model stacking (adding result of one or more models as a datapoint for the next model)
    # fill all NaNs with column-wise mean
    df.views.fillna(df.views.mean(), inplace=True)
    df.imdb_rating.fillna(df.imdb_rating.mean(), inplace=True)
    df.major_char_lines.fillna(df.major_char_lines.mean(),inplace=True)
    df.max_line_length.fillna(df.max_line_length.mean(),inplace=True)
    df.line_lengths.fillna(df.line_lengths.mean(),inplace=True)
    df.locations_in_ep.fillna(df.locations_in_ep.mean(), inplace=True)
    # get X and y
    y = df['imdb_rating']
    # X = df[['id', 'season', 'number_in_season', 'number_in_series', 'views', 'title_len', 'election_year', 'word_count', 'homer_and_bart']]
    X = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']]
    # build stacked model using Gradient Boost, AdaBoost, and Random Forest
    # train test split (training , hold out)
    training_x, hold_out_x, training_y, holdout_y = tts(X, y, test_size = .25)
    # > RANDOM FOREST
    # train random forest on training set using cross validation
    print('\n')
    print('training rfr')
    print('\n')
    _rfr = r_fr(n_estimators=int(trees * .2), n_jobs=-1, verbose=False, min_impurity_split=1e-6)
    score_rfr = sqrt(abs(np.array(cvs(_rfr, training_x, training_y, cv=5, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    _rfr.fit(training_x, training_y)
    print('Random_forest_score = ', score_rfr)
    # predict values for all of trainig set and pass back into X
    training_x['rfr_score'] = _rfr.predict(training_x)
    # ADABOOST REGRESSOR
    # train adaboost with X and the scores from rfr using k_fold
    print('\n')
    print('training abr')
    print('\n')
    _abr = a_br(n_estimators=trees, learning_rate=.4, loss = 'exponential')
    score_abr = sqrt(abs(np.array(cvs(_abr, training_x, training_y, cv=5, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    _abr.fit(training_x, training_y)
    print('Adaboost_score = ', score_abr)
    # # predict all training values and pass back into X
    training_x['abr_score'] = _abr.predict(training_x)
    # > GRADIENT BOOSTED REGRESSOR
    # train gradient boosted model on all of X with rfr and abr scores
    print('\n')
    print('training gbr')
    print('\n')
    _gbr = g_br(loss = 'lad', learning_rate = .1, n_estimators=trees, subsample = .85, warm_start=False, verbose=False)
    score_gbr = sqrt(abs(np.array(cvs(_gbr, training_x, training_y, cv=5, n_jobs = -1, verbose = True, scoring = 'neg_mean_squared_error')).mean()))
    _gbr.fit(training_x, training_y)
    print('Gradient_boosted_score_cross_val_score = ', score_gbr)
    # test on hold out by passing holdout data through all three models sequentially
    print(list(zip(_gbr.feature_importances_, ['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep', 'rfr_score', 'abr_score'])))
    return _rfr, _abr, _gbr, hold_out_x, holdout_y

def predict_on_holdout(_rfr, _abr, _gbr, x_h, y_h):
    x_h['rfr_score'] = _rfr.predict(x_h)
    x_h['abr_score'] = _abr.predict(x_h)
    preds = _gbr.predict(x_h)
    x_h['stacked_score'] = preds
    rmse = sqrt(mse(y_h, preds))
    x_h['rating'] = y_h
    plot_scores(x_h, rmse)
    return rmse

def plot_scores(df, rmse):
    df = df.sort_values('id')
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
    _rfr, _abr, _gbr, x_h, y_h = stack_models(episode_df)
    print('stacked model score = ', predict_on_holdout(_rfr, _abr, _gbr, x_h, y_h))

# side note - I want to add in these details from scripts:
# song (bool)
# monologue (line len > 90 words)
