import pandas as pd
import numpy as np
from sklearn.externals import joblib
import filtering as fe


def make_dict(char_df, loc_df, vals, song, politics):
    new_dict1, new_dict2 = dict(), dict()
    count = 1
    for character in char_df:
        for locat in loc_df:
            for val in vals:
                for m in song:
                    for p in politics:
                        new_dict1[count] = [character, locat, val, m, p]
                        new_dict2[count] = fe.return_suggested([character, locat, val, m, p])
                        count += 1
                        print('done')
    joblib.dump(new_dict1, 'hash_table_test.pkl')
    joblib.dump(new_dict2, 'hash_table_test2.pkl')

if __name__ == '__main__':
    char_df = pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_characters.csv').index
    loc_df = pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_locations.csv').index
    vals = ['Characters', 'Location']
    song = [True, False]
    politics = [True, False]
    make_dict(char_df, loc_df, vals, song, politics)
