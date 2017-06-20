import pandas as pd
import numpy as np
from sklearn.externals import joblib
import filtering as fe


def make_dict(char_df, loc_df, vals, song, politics):
    '''
    INPUT: five items to make prediction on
    OUTPUT: None - pickled hash table (dictionary) with results
    '''
    # initialize the dictionaries
    new_dict1, lst = dict(), []
    default_video = 'http://www.simpsonsworld.com/video/288056387698'
    default_image = 'http://static-media.fxx.com/img/FX_Networks_-_FXX/925/395/Simpsons_05_08_P2.jpg'
    for character in char_df:
        for locat in loc_df:
            for val in vals:
                for m in song:
                    for p in politics:
                        group = [character, locat, val, m, p]
                        print(group)
                        if ' '.join(str(x) for x in group) in set(new_dict1.keys()):
                            print('skipped')
                            pass
                        else:
                            print('here')
                            y_hat = fe.return_suggested(group)
                            if y_hat:
                                print('adding')
                                new_dict1[' '.join(str(x) for x in group)] = y_hat
                            else:
                                print('adding default')
                                new_dict1[' '.join(str(x) for x in group)] = [
                                    "Boy-Scoutz 'n the Hood",
                                    8.56, 8.4, default_image, default_video
                                    ]
    # pickle the hash table
    joblib.dump(new_dict1, 'hash_table_test.pkl')

if __name__ == '__main__':
    # get all options for the user input in Flask App
    char_df = pd.read_csv('data/simpsons_characters.csv').index
    loc_df = pd.read_csv('data/simpsons_locations.csv').index
    vals = ['Characters', 'Location']
    song = [True, False]
    politics = [True, False]
    # pass those into make_dict function
    make_dict(char_df, loc_df, vals, song, politics)
