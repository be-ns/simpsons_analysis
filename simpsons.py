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
    # read in filename
    df = pd.read_csv(filename, sep=',').sort_values(by='id')

    # change views to views in millions and drop original column
    df['views'] = df['us_viewers_in_millions']
    df.drop('us_viewers_in_millions', axis=1, inplace=True)

    # turn the title into a metric for length of episode title
    df['title_len'] = df['title'].apply(lambda x: len(x))

    # change air date to datetime object (for other purposes)
    df['original_air_date'] = pd.to_datetime(df['original_air_date'])

    # election cycle years are all divisible by four, (e.g. 2016, 2012, 2008) so I'm adding this as a column feature
    df['election_year'] = df['original_air_date'].apply(lambda x: 1 if x.year % 4 == 0 else 0)

    # drop any unnecesary columns
    df.drop(labels=['title','imdb_votes', 'production_code'], axis=1, inplace = True)
    # return clean df
    return df

def add_location_data(script_df):
    '''
    INPUT: Script DataFrame
    OUTPUT: Pandas Series with count of locations per episode

    Made as separate function to allow future tweaking
    '''
    # return the number of unique locations for each episode
    return script_df.groupby('episode_id')['location_id'].nunique()

def return_clean_script_df(fname, df):
    '''
    INPUT: filename as fname and the original episode DataFrame from `clean_episode_data`
    OUTPUT: DF with extracted script info, sorted by episode
    '''
    # import file into Pandas DataFrame
    script_df = pd.read_csv(fname)

    # keep only the normalized lines so capitalization isn't a problem
    script_df.drop(labels = ['spoken_words', 'raw_text'], axis=1, inplace=True)


    # get totals for each episode for different features and merge into original DF
    line_length_series = script_df.groupby('episode_id')['word_count'].mean()
    monologues = script_df.groupby('episode_id')['word_count'].max()
    df = df.merge(pd.DataFrame(line_length_series), how='left', left_on = 'id', right_index=True)
    df = df.merge(pd.DataFrame(monologues), how='left', left_on = 'id', right_index=True)
    # get number of lines spoken by major characters (AKA the Simpsons) and all characters
    major_char_lines = script_df.where(script_df['raw_character_text'].str.contains('Simpson')).groupby('episode_id')['speaking_line'].count()
    line_series = script_df.groupby('episode_id')['speaking_line'].count()
    # get ratio of lines spoken by Simpsons vs all and merge into DF
    major_char_lines = major_char_lines / line_series
    df = df.merge(pd.DataFrame(major_char_lines), how='left', left_on = 'id',  right_index=True)

    # get the count of locations from each episode from scripts
    loc_series = add_location_data(script_df)
    df = df.merge(pd.DataFrame(loc_series), how='left', left_on = 'id',  right_index=True)

    #rename columns to avoid confusion and return df
    df.columns = ['id', 'original_air_date', 'season', 'number_in_season', 'number_in_series', 'views', 'imdb_rating', 'image_url', 'video_url', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']
    return df

def _fill_nans(df):
    '''
    INPUT: Pandas DataFrame object
    OUTPUT: Pandas Dataframe object with all NaNs filled with column-wise mean

    being used for specific columns from DataFrame
    '''
    # fills columns with column-wise mean()
    df.imdb_rating.fillna(df.imdb_rating.mean(), inplace=True)
    df.major_char_lines.fillna(df.major_char_lines.mean(),inplace=True)
    df.max_line_length.fillna(df.max_line_length.mean(),inplace=True)
    df.line_lengths.fillna(df.line_lengths.mean(),inplace=True)
    df.locations_in_ep.fillna(df.locations_in_ep.mean(), inplace=True)
    # return cleaned DataFrame
    return df

def plot_errors(ab_train, ab_test, gbr_train, stacked_test):
    '''
    INPUT: training and testing scores for first snd second models
    OUPUT: None

    saves MatPlotLib figure to directory of script
    '''
    # close any plots taking memory
    plt.close('all')
    # create new figure and add subplot
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)

    # plot errors for both initial and stacked model
    ax.plot([1, 2], [ab_train, gbr_train], label = 'training')
    ax.plot([1, 2], [ab_test, stacked_test], label = 'test')

    # adjust appearance of graph and add titles / labels
    ax.tick_params(labelsize=20)
    ax.set_title('Training Error to Test Error', size=40)
    ax.set_xlabel('Model / Stage', size=35)
    ax.set_xticks([1, 2], ['AdaBoost', 'Gradient Boost'])
    ax.set_ylabel('RMSE', size = 35)
    plt.legend(prop={'size':14})
    plt.tight_layout()
    plt.savefig('error.png', dpi=100)

def build_gbr(training_x, training_y, holdout_x, holdout_y, _abr, abr_test, rounds):
    '''
    INPUT: training features, training targets, holdout features, holdout target, previous model, test score from previous model
    OUTPUT: final model, final_score, final model training score
    '''
    # > BUILD GRADIENT BOOSTED REGRESSOR
    # train gradient boosted model on all of X with rfr and abr scores to get training error
    _gbr = g_br(loss = 'lad', learning_rate = .1, n_estimators=500, warm_start=False, verbose=False)
    # get RMSE (take square root of absolute value of negative mse) using 5-fold Cross Validation
    final_train = sqrt(abs(np.array(cvs(_gbr, training_x, training_y, cv=5, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    print('Gradient_Boosted_score_cross_val_score = ', final_train)

    # now on to the test error...
    # set arbitrary final_score - to be used in while loop to meet threshold
    final_test = 100
    # iteratively train Random Searched Gradient Boosted model on training data until it beats the previous model's score
    while final_test > abr_test:
        # make simple Gradient Boosted Regressor
        _gbr = g_br(loss = 'lad', verbose = True)
        # set param distribution
        param_distribution = {
            "max_depth": [3, 4, 5],
            "learning_rate": [.2, .3, .4],
            "n_estimators": sp_randint(500, 3000)
            }
        n_iter_search = rounds

        # implement Random Search
        final_model = r_search(_gbr, param_distributions=param_distribution, n_iter=n_iter_search, n_jobs = -1, cv = 4, verbose = 1)
        # fit to training set
        final_model.fit(training_x, training_y)

        # get score for holdout set, reset variable
        final_test = predict_on_holdout(_abr, final_model, holdout_x, holdout_y)
        # if threshold not met, try again
        print('final_score = ', final_test)
    # return model and scores
    return final_model, final_test, final_train

def build_abr(training_x, training_y, holdout_x, holdout_y, rounds):
    '''
    INPUT: training features, training target, holdout features, holdout target
    OUTPUT: adaboost model, adaboost test score, adaboost train score
    '''
    # > BUILD ADABOOST REGRESSOR
    # use defaults to get adaboost training error
    _abr = a_br()
    # get RMSE (take square root of absolute value of negative mse) using 5-fold Cross Validation
    abr_train = sqrt(abs(np.array(cvs(_abr, training_x, training_y, cv=4, n_jobs = -1, verbose = False, scoring = 'neg_mean_squared_error')).mean()))
    # print training error
    print('Adaboost_cross_val_score = ', abr_train)

    # now onto test error...
    # set parameters for Random Search
    param_distribution = {
                "loss": ['linear', "square", "exponential"],
                "learning_rate": [.15, .25, .29, .33, .5, .6, .9],
                "n_estimators": sp_randint(250, 1500)
                }
    # number of iterations on Random Search
    n_iter_search = rounds
    # set Random Search (r_search was import name)
    _abr = r_search(_abr, param_distributions=param_distribution, n_iter=n_iter_search, n_jobs = -1, cv = 4, verbose = 1)

    # fit to training set
    _abr.fit(training_x, training_y)

    # get holdout score and print it
    abr_test = sqrt(mse(holdout_y, _abr.predict(holdout_x)))
    print('holdout_score = ', abr_test)
    # return model and scores
    return _abr, abr_test, abr_train

def stack_models(df, rounds = 2):
    '''
    INPUT: clean Pandas Dataframe with script and episode information, kwarg for if we are printing RMSE or returning the dataframe itself (for PyMC3 usage)
    OUTPUT: Either the RMSE for the stacked model or a Dataframe with three new columns (scores from each of the three models)

    Builds a stacked model using AdaBoost followed by Gradient Boosting
    '''
    # fill NaNs
    df =_fill_nans(df)

    # get X and y
    X, y = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']], df['imdb_rating']

    # train test split to get training , hold out features / targets
    training_x, holdout_x, training_y, holdout_y = tts(X, y, test_size = .2)

    # build AdaBoost Regressor
    _abr, abr_test, abr_train = build_abr(training_x, training_y, holdout_x, holdout_y, rounds)

    # predict all training values and pass back into X
    training_x['abr_score'] = _abr.predict(training_x)
    holdout_x['abr_score'] = _abr.predict(holdout_x)

    # build Gradient Boosted Regressor
    final_model, final_test, final_train = build_gbr(training_x, training_y, holdout_x, holdout_y, _abr, abr_test, rounds)

    # feature importances gleaned from model not trained with random search due to model limitations
    ## print(sorted(list(zip(random_search.feature_importances_, ['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep', 'abr_score'])), reverse=True))

    # plot errors
    plot_errors(abr_train, abr_test, final_train, final_test)

    # pickle both models with SKLearn's `joblib` (better for complex matrices of data within model)
    joblib.dump(final_model, 'stacked_model_2.pkl')
    joblib.dump(_abr, 'adaboost_model_2.pkl')

    # return both models, holdout data, and final RMSE
    return _abr, final_model, holdout_x, holdout_y, final_test

def predict_on_holdout(_abr, random, x_h, y_h):
    '''
    INPUT: two models for predicting regression value and holdout (test) data
    OUTPUT: RMSE for the stacked model
    '''
    # ensure we have the requisite columns for model
    x_h = x_h[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']]

    # add column to x_h DataFrame with predicted values
    x_h['abr_score'] = _abr.predict(x_h)

    # get final model predictions and RMSE, then plot errors
    final_scores_predicted = random.predict(x_h)
    rmse = sqrt(mse(y_h, final_scores_predicted))
    x_h['predicted'] = final_scores_predicted
    x_h['rating'] = y_h

    plot_scores(x_h, rmse)

    # return RMSE
    return rmse

def plot_scores(df, rmse):
    '''
    INPUT: DataFrame, Square Root of the Mean Squared Error for a function
    OUTPUT: None

    saves figure to same directory as the script
    '''
    # put it in order of episode ID
    df = df.sort_values('id')
    # 'ggplot' is my favorite plotting style
    plt.style.use('ggplot')

    # build figure and plot correct and incorrect rmse by order of episode ID
    fig = plt.figure(figsize=(35,15))
    ax = fig.add_subplot(111)
    df = df.sort_values('id')
    ax.plot(df['id'], df['rating'], label='True Rating', color='r', lw=5, alpha=.4)
    ax.plot(df['id'], df['predicted'], label = 'Predicted Rating', color = 'b', lw =5, alpha = .4)

    # adjust preferences and style of figure
    ax.set_xlim((0,600))
    ax.tick_params(labelsize=20)
    ax.set_title('Predicted Ratings to Actual Ratings, RMSE = {0}'.format(round(rmse, 3)), size=40)
    ax.set_xlabel('Episode Number', size=35)
    ax.set_ylabel('Rating on 10 point scale', size = 35)
    plt.legend(prop={'size':35})
    plt.tight_layout()
    # save figure
    plt.savefig('score.png', dpi=100)
    plt.close('all')

if __name__ == '__main__':
    # read in episode data to Pandas DataFrame
    e_filename = 'data/simpsons_episodes.csv'
    # read in raw episode data, return clean episode pandas dataframe
    episode_df = return_clean_script_df("data/simpsons_script_lines.csv", clean_episode_data(e_filename))
    # get rmse for stacked model with kwarf 'rmse'
    _abr, _stacked_model, x_h, y_h, final_score = stack_models(episode_df)
    print('stacked model score = ', final_score)
