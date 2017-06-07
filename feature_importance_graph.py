import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def plot_feats(nums, chars):
    '''
    INPUT: list of tuples with feature importance as index 0 and feature name as index 1
    OUTPUT: saves plot to same directory as script, nothing is returned
    '''
    objects = set(chars)
    y_pos = np.arange(len(objects))
    performance = nums
    plt.bar(y_pos, performance, align='center', alpha=0.9, color='#1414B4')
    plt.xticks(y_pos, objects, rotation=45)
    plt.ylabel('Importance')
    plt.title('Top Features in Stacked model')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=100)

def divide_feats():
    imp_feats, nums, chars = [(0.1768, 'line_lengths'), (0.1664, 'major_char_lines'), (0.1482, 'abr_score'), (0.1330, 'max_line_length'), (0.1261, 'id'), (0.1038, 'title_len'), (0.0806, 'locations_in_ep'), (0.0583, 'number_in_season'), (0.0065, 'election_year')], [], []
    for tup in imp_feats:
        nums.append(tup[0])
        chars.append(tup[1])
    return nums, chars

if __name__ == '__main__':
    plt.style.use('ggplot')
    nums, chars = divide_feats()
    plot_feats(nums, chars)
