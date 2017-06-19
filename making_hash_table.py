import pandas as pd
import numpy as np
from sklearn.externals import joblib
import filtering as fe


def make_dict(char_df, loc_df, vals, song, politics):
    new_dict1, new_dict2 = dict(), dict()
    lst = []
    count = 1
    for character in char_df:
        for locat in loc_df:
            for val in vals:
                for m in song:
                    for p in politics:
                        lst.append([character, locat, val, m, p])
        for group in lst:
            print(group)
            x = fe.return_suggested(group)
            if x:
                new_dict1[(character, locat, val, m, p)] = x
                count += 1
            else:
                new_dict1[(character, locat, val, m, p)] = ["Boy-Scoutz 'n the Hood",8.56,8.4, 'http://static-media.fxx.com/img/FX_Networks_-_FXX/280/1003/Simpsons_05_08.jpg', 'http://www.simpsonsworld.com/video/288056387698']
                count += 1
    # pickle the models
    joblib.dump(new_dict1, 'hash_table_test.pkl')
    joblib.dump(new_dict2, 'hash_table_test2.pkl')

if __name__ == '__main__':
    char_df = pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_characters.csv').index
    loc_df = pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_locations.csv').index
    vals = ['Characters', 'Location']
    song = [True, False]
    politics = [True, False]
    make_dict(char_df, loc_df, vals, song, politics)
