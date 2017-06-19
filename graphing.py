import numpy as np
from sklearn.externals import joblib
import simpsons as s
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as _rfr, AdaBoostRegressor as _abr, GradientBoostingRegressor as _gbr
from sklearn.neighbors import KNeighborsRegressor as _knn
from sklearn.neural_network import MLPRegressor as _mlp
from sklearn.metrics import mean_squared_error as mse
from math import sqrt


def initialize():
    '''
    INPUT: None
    OUTPUT: None - calls
    '''
    abm = joblib.load("pickled_models/adaboost_model.pkl")
    gbm = joblib.load("pickled_models/stacked_model.pkl")
    # read in episode data to Pandas DataFrame
    e_filename = 'data/simpsons_episodes.csv'
    # read in raw episode data, return clean episode pandas dataframe
    episode_df = s.return_clean_script_df("data/simpsons_script_lines.csv", s.clean_episode_data(e_filename))
    # fill NaNs
    df = s._fill_nans(episode_df)
    # get X and y (features and target)
    X, y = df[['id', 'number_in_season', 'title_len', 'election_year', 'line_lengths', 'max_line_length', 'major_char_lines', 'locations_in_ep']], df['imdb_rating']
    __train_make_predictions(X, y, abm, gbm)

def __train_make_predictions(X, y, abm, gbm):
    '''
    INPUT: feature matrix (X), target Series (y), two models for final stack
    OUTPUT: None - function makes graphs
    '''
    # create several models for comparison
    knn, rfr, abr, gbr = _knn(), _rfr(n_estimators = 400), _abr(n_estimators = 400), _gbr(n_estimators = 400)
    # list for errors
    training_errors =[]
    testing_errors = []
    # train_test_split
    x_tr, x_te, y_tr, y_te = tts(X, y)
    # fit several models to the feature matrix
    knn.fit(x_tr, y_tr)
    rfr.fit(x_tr, y_tr)
    abr.fit(x_tr, y_tr)
    gbr.fit(x_tr, y_tr)
    # get predictions for the training (for training error)
    k_tr = knn.predict(x_tr)
    r_tr = rfr.predict(x_tr)
    a_tr = abr.predict(x_tr)
    g_tr = gbr.predict(x_tr)
    x_tr['abr_score'] = abm.predict(x_tr)
    stacked_tr = gbm.predict(x_tr)
    models = [k_tr, r_tr, a_tr, g_tr, stacked_tr]
    for model in models:
        # add error to a list to show differences
        training_errors.append(sqrt(mse(y_tr, model)))
    # get predictions for the testing (for training error)
    k_te = knn.predict(x_te)
    r_te = rfr.predict(x_te)
    a_te = abr.predict(x_te)
    g_te = gbr.predict(x_te)
    x_te['abr_score'] = abm.predict(x_te)
    stacked_te = gbm.predict(x_te)
    models = [k_te, r_te, a_te, g_te, stacked_te]
    for model in models:
        # add errors to a list for plotting
        testing_errors.append(sqrt(mse(y_te, model)))
    # plot the errors (train and test)
    _plot_train_test_errors(models, training_errors, testing_errors)

def _plot_train_test_errors(models, training_errors, testing_errors):
    # set number of columns for the plot
    n_models = len(training_errors)
    # add subplots
    fig, ax = plt.subplots()
    # make a range for number of groups
    index = np.arange(n_models)
    # set bar width
    bar_width = 0.35
    opacity = 0.7
    error_config = {'ecolor': '0.3'}
    # plot errors for training
    rects1 = plt.bar(index, training_errors, bar_width,
                     alpha=opacity,
                     color='#FF8B00',
                     error_kw=error_config,
                     label='TRAINING')
    # plot error for test
    rects2 = plt.bar(index + bar_width, testing_errors, bar_width,
                     alpha=opacity,
                     color='#0068FF',
                     error_kw=error_config,
                     label='TESTING')
    # set the graph features
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Training / Testing Error by model')
    plt.xticks(index + bar_width / 2, ('KNN', 'Random Forest', 'AdaBoost Tree', 'Gradient Boosted Tree', 'Final Stacked Model'),  rotation=40)
    plt.legend()
    plt.tight_layout()
    plt.savefig('error_rates.png', dpi=300)

def _plot_x(X, seasons):
    '''
    INPUT: Feature Matrix with episode ID (X), seasons as a np.array (seasons)
    OUTPUT: None - saves plot of predicted scores to actual scores for all  seasons of TV show
    '''
    # get only ID
    X = X.sort_values('id')
    # Make Plot
    fig = plt.figure(figsize=(25, 20))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # plot the first few seasons
    ax1.plot(X.id[:276:4], X.pred[:276:4], alpha = .9, lw=4, label='PREDICTED RATINGS', c='#0068FF')
    # plot the next few seasons
    ax1.plot(X.id[:276:4], X.true[:276:4], alpha = .6, lw=12, label='TRUE RATINGS', c='#FF8B00')
    # set plot parameters
    ax1.set_xlim((0,276))
    ax1.set_ylim((4.5, 10))
    ax1.set_xticks(np.arange(0,276, 24)) # number
    ax1.set_xticklabels(np.arange(1, 13))
    ax1.tick_params(labelsize=35)
    ax1.set_title('Predicted Ratings to Actual Ratings, RMSE = .351', size=50)
    ax1.set_xlabel('Season', size = 45)
    ax1.set_ylabel('Rating on 10 point scale', size = 45)
    ax1.legend(prop={'size':30})
    ax2.plot(X.id[276:600:4], X.pred[276:600:4], alpha = .9, lw=4, label='PREDICTED RATINGS', c='#0068FF')
    ax2.plot(X.id[276:600:4], X.true[276:600:4], alpha = .6, lw=12, label='TRUE RATINGS', c='#FF8B00')
    ax2.set_xlim((276,600))
    ax2.set_ylim((4.5, 10))
    ax2.set_xticks(np.arange(276, 600, 24))
    ax2.set_xticklabels(np.arange(12, 26))
    ax2.tick_params(labelsize=35)
    ax2.set_title('Predicted Ratings to Actual Ratings, RMSE = .351', size=50)
    ax2.set_xlabel('Season', size = 45)
    ax2.set_ylabel('Rating on 10 point scale', size = 45)
    ax2.legend(prop={'size':30})
    plt.tight_layout()
    # save figure
    plt.savefig('score.png', dpi=300)


if __name__ == '__main__':
    # set plot type to ggplot
    plt.style.use('ggplot')
    initialize()
