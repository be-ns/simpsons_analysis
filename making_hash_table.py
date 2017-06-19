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
    new_dict1, new_dict2, lst, count = dict(), dict(), [], count = 1
    default_video = 'http://www.simpsonsworld.com/video/288056387698'
    default_image =
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
                new_dict1[(character, locat, val, m, p)] = [
                                        "Boy-Scoutz 'n the Hood",
                                        8.56, 8.4, default_image, default_video
                                        ]
                count += 1
    # pickle the hash table
    joblib.dump(new_dict1, 'hash_table_test.pkl')
    joblib.dump(new_dict2, 'hash_table_test2.pkl')

if __name__ == '__main__':
    # get all options for the user input in Flask App
    char_df = pd.read_csv('data/simpsons_characters.csv').index
    loc_df = pd.read_csv('data/simpsons_locations.csv').index
    vals = ['Characters', 'Location']
    song = [True, False]
    politics = [True, False]
    # pass those into make_dict function
    make_dict(char_df, loc_df, vals, song, politics)
