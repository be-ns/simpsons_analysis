import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor as g_br, RandomForestRegressor as r_fr, AdaBoostRegressor as a_br
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor as k_nn
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs, RandomizedSearchCV as r_search
from scipy.stats import randint as sp_randint

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

def _fill_nans(df):
    df.imdb_rating.fillna(df.imdb_rating.mean(), inplace=True)
    df.major_char_lines.fillna(df.major_char_lines.mean(),inplace=True)
    df.max_line_length.fillna(df.max_line_length.mean(),inplace=True)
    df.line_lengths.fillna(df.line_lengths.mean(),inplace=True)
    df.locations_in_ep.fillna(df.locations_in_ep.mean(), inplace=True)
    return df

def plot_errors(ab_train, ab_test, gbr_train, stacked_test):
    plt.close('all')
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    ax.plot([1, 2], [ab_train, gbr_train], label = 'training')
    ax.plot([1, 2], [ab_test, stacked_test], label = 'test')
    # ax.set_xlim((0,600))
    ax.tick_params(labelsize=20)
    ax.set_title('Training Error to Test Error', size=40)
    ax.set_xlabel('Model / Stage', size=35)
    ax.set_xticks([1, 2], set('AdaBoost', 'Gradient Boost'), rotation=45)
    ax.set_ylabel('RMSE', size = 35)
    plt.legend(prop={'size':14})
    plt.tight_layout()
    plt.savefig('error.png', dpi=100)

def build_gbr(training_x, training_y, abr_test, holdout_x, holdout_y, _abr, trees = 5000):
    # > BUILD GRADIENT BOOSTED REGRESSOR
    # train gradient boosted model on all of X with rfr and abr scores
    _gbr = g_br(loss = 'lad', learning_rate = .1, n_estimators=200, warm_start=False, verbose=False)
    gbr_train = sqrt(abs(np.array(cvs(_gbr, training_x, training_y, cv=5, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    print('Gradient_Boosted_score_cross_val_score = ', gbr_train, 'trees = {0}'.format(trees))
    final_score = 100
    while final_score > abr_test:
        # get RMSE (take square root of absolute value of negative mse)
        _gbr = g_br(loss = 'lad', verbose = True)
        param_distribution = {
            "max_depth": [3, 4, 5],
            "learning_rate": [.2, .3, .4],
            "n_estimators": sp_randint(500, 3000)
            }
        n_iter_search = 30
        random_search = r_search(_gbr, param_distributions=param_distribution,
                                   n_iter=n_iter_search, n_jobs = -1, cv = 4, verbose = 1)
        # fit to training set
        random_search.fit(training_x, training_y)

        # get holdout score
        final_score = predict_on_holdout(_abr, random_search, holdout_x, holdout_y)
        print('final_score = ', final_score)

        # once threshold is met, get feature importances and plot errors
    return random_search, final_score, gbr_train

def build_abr(training_x, training_y, holdout_x, holdout_y, trees = 1000):
    # > BUILD ADABOOST REGRESSOR
    # train adaboost with new X using 5-fold Cross Validation
    _abr = a_br()
    # get RMSE (take square root of absolute value of negative mse)
    abr_train = sqrt(abs(np.array(cvs(_abr, training_x, training_y, cv=4, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    print('Adaboost_cross_val_score = ', abr_train)
    param_distribution = {
                "loss": ['linear', "square", "exponential"],
                "learning_rate": [.15, .25, .29, .33, .5, .6, .9],
                "n_estimators": sp_randint(250, 1500)
                }
    n_iter_search = 30
    random_search = r_search(_abr, param_distributions=param_distribution, n_iter=n_iter_search, n_jobs = -1, cv = 4, verbose = 1)
            # fit to training set
    random_search.fit(training_x, training_y)
    # get holdout score
    abr_test = sqrt(mse(holdout_y, random_search.predict(holdout_x)))
    print('holdout_score = ', abr_test)
    return random_search, abr_test, abr_train

def stack_models(df, trees = 1000):
    '''
    INPUT: clean Pandas Dataframe with script and episode information, kwarg for if we are printing RMSE or returning the dataframe itself (for PyMC3 usage)
    OUTPUT: Either the RMSE for the stacked model or a Dataframe with three new columns (scores from each of the three models)
    '''
    df = _fill_nans(df)
    # get X and y
    X, y = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']], df['imdb_rating']

    # build stacked model using Gradient Boost, AdaBoost, and Random Forest

    # train test split to get training , hold out
    training_x, holdout_x, training_y, holdout_y = tts(X, y, test_size = .2)

    _abr, abr_test, abr_train = build_abr(training_x, training_y, holdout_x, holdout_y)

    # predict all training values and pass back into X
    training_x['abr_score'] = _abr.predict(training_x)
    holdout_x['abr_score'] = _abr.predict(holdout_x)
    random_search, final_score, gbr_train = build_gbr(training_x, training_y, abr_test, holdout_x, holdout_y, _abr)
    if final_score > .430 or final_score > abr_test:
        stack_models(df)
    # print(sorted(list(zip(random_search.feature_importances_, ['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep', 'abr_score'])), reverse=True))
    plot_errors(abr_train, abr_test, gbr_train, final_score)
    joblib.dump(random_search, 'stacked_model_2.pkl')
    joblib.dump(_abr, 'adaboost_model_2.pkl')
    return _abr, random_search, holdout_x, holdout_y, final_score

def predict_on_holdout(_abr, random, x_h, y_h):
    x_h = x_h[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']]
    x_h['abr_score'] = _abr.predict(x_h)
    preds = random.predict(x_h)
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
    ax.plot(df['id'], df['rating'], label='True Rating', color='r', lw=5, alpha=.4)
    ax.plot(df['id'], df['stacked_score'], label = 'Predicted Rating', color = 'b', lw =5, alpha = .4)
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
    _abr, _gbr, x_h, y_h, final_score = stack_models(episode_df)
    # model = joblib.load('/Users/benjamin/Desktop/DSI/simpsons_analysis/stacked_model.pkl')
    print('stacked model score = ', final_score)
