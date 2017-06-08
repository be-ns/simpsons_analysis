import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def plot_feats():
    '''
    INPUT: list of tuples with feature importance as index 0 and feature name as index 1
    OUTPUT: saves plot to same directory as script, nothing is returned
    '''
    nums, chars = divide_feats()
    y_pos = np.arange(len(chars))
    plt.bar(y_pos, nums, align='center', alpha=0.9, color='#1414B4')
    plt.xticks(y_pos, chars, rotation=80)
    plt.ylabel('Pct Variance Explained')
    plt.title('Top Features in Stacked model')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=100)

def divide_feats():
    imp_feats = np.array([[0.1768, 'line_lengths'], [0.1664, 'major_char_lines'], [0.1482, 'abr_score'], [0.1330, 'max_line_length'], [0.1261, 'id'], [0.1038, 'title_len'], [0.0806, 'locations_in_ep'], [0.0583, 'number_in_season'], [0.0065, 'election_year']])
    return imp_feats[:,0], ['average line length', 'Simpsons to Others: speaking lines', 'AdaBoost Prediction', 'Longest Line (monologue)', 'show id', 'Length of Title', 'Number of Locations', 'Episode Number in Season', 'Election year or Not']

def plot_loss():
    test = [.44325, .41225]
    train = [.47625, .42825]
    models = ['','Adaboost Only', '', 'Gradient Boosted and Adaboost','']
    y_pos_2 = [1,2]
    plt.plot(y_pos_2, test, alpha=0.8, color='#FAB812', label = 'test error rate', lw = 3)
    plt.plot(y_pos_2, train, alpha=0.8, color='#1414B4', label = 'train error rate', lw = 3)
    plt.xticks([.5, 1, 1.5, 2, 2.5], models, rotation=10)
    plt.xlim((.95, 2.05))
    plt.ylabel('Root Mean Squared Error', size=15)
    plt.title('Reduction in RMSE for stacked model: CV = 4', size=16)
    plt.xlabel('Model', size = 10)
    plt.legend(prop={'size':10})
    plt.tight_layout()
    plt.savefig('reduction_in_rmse.png', dpi=100)

if __name__ == '__main__':
    plt.style.use('ggplot')
    plot_feats()
    # plot_loss()
    plt.close('all')
