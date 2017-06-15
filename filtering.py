import pandas as pd
import simpsons as s
import numpy as np
from string import punctuation
import loading_data as ld


def filter_df(char, scripts = "data/simpsons_script_lines.csv", episodes = "data/simpsons_episodes.csv"):
    '''
    INPUT: filepaths for script data and episode data (defaulted)
    OUTPUT: Pandas DataFrame with ratio of speaking lines for user inputted character for every episode

    NaNs are filled after ratio for minor characters so any episodes without the character included would be eliminated
    '''
    # read in filepaths
    script_df = pd.read_csv(scripts)
    episode_df = pd.read_csv(episodes)
    # ensure all character names are lowercased
    script_df['raw_character_text'] = np.array([str(x).lower() for x in script_df.raw_character_text])
    # find the chosen characters lines
    char_series = script_df['raw_character_text'].apply(lambda x: 1 if char.lower().strip() in str(x) else 0)
    script_df['char_lines'] = char_series

    # find all other character lines summed lines
    otherlines = script_df['raw_character_text'].apply(lambda x: 1 if not char.lower().strip() in str(x) else 0)
    script_df['non_char_lines'] = otherlines
    script_df = script_df.groupby('episode_id', as_index = False)[['char_lines', "non_char_lines"]].sum()
    # make char_ratio
    script_df['char_ratio'] = (script_df['char_lines'] / script_df['non_char_lines'])
    # merge that into the episode DF and return it
    merged_df = episode_df.merge(script_df, how='left', left_on = 'id', right_on='episode_id')
    return merged_df, scripts

def filter_locations(episode_df, location, scripts):
    script_df = pd.read_csv(scripts)
    script_df['raw_location_text'] = np.array([str(x).lower() for x in script_df.raw_location_text])

    loc_series = script_df['raw_location_text'].apply(lambda x: 1 if location.lower().strip() in str(x) else 0)
    script_df['loc_lines'] = loc_series

    # find all other location lines summed lines
    otherlines = script_df['raw_location_text'].apply(lambda x: 1 if not location.lower().strip() in str(x) else 0)
    script_df['non_loc_lines'] = otherlines
    script_df = script_df.groupby('episode_id', as_index = False)[['loc_lines', "non_loc_lines"]].sum()
    # make loc_ratio
    script_df['loc_ratio'] = (script_df['loc_lines'] / script_df['non_loc_lines'])
    # merge that into the episode DF and return it
    merged_df = episode_df.merge(script_df, how='left', left_on = 'id', right_on='episode_id')
    return merged_df

def get_song(episode_df, scripts, song):
    script_df = pd.read_csv(scripts)
    if song:
        script_df['song'] = script_df['raw_text'].apply(lambda x: 1 if ('singing' in str(x).lower() or 'sing' in str(x).lower()) else 0)
        # `sie` to stand for singers in episode
        script_df['sie'] = script_df['raw_character_text'].apply(lambda x: 1 if 'singer' in str(x).lower() else 0)
    elif song == False:
        script_df['song'] = script_df['raw_text'].apply(lambda x: 0 if ('singing' in str(x).lower() or 'sing' in str(x).lower()) else 1)
        # `sie` to stand for singers in episode
        script_df['sie'] = script_df['raw_character_text'].apply(lambda x: 0 if 'singer' in str(x).lower() else 1)\

    script_df['song'] = script_df['song'] + script_df['sie']
    script_df.drop('sie', axis = 1, inplace = True)
    script_df = script_df.groupby("episode_id", as_index = False)['song'].sum()

    return episode_df.merge(script_df, how='left', left_on = 'id', right_on = 'episode_id')


def politicize(episode_df, scripts, politics):
    script_df = pd.read_csv(scripts)
    politics_lst = 'president politics political election congress right-wing immigration environment campaign clinton obama cheney'.lower().split(' ')
    for word in politics_lst:
        if politics == True:
            for word in politics_lst:
                script_df[word] = script_df['raw_text'].apply(lambda x: 1 if word in str(x).lower() else 0)
        elif politics == False:
            for word in politics_lst:
                script_df[word] = script_df['raw_text'].apply(lambda x: 0 if word in str(x).lower() else 1)
    script_df['politics'] = script_df.president + script_df.politics +  script_df.political +  script_df.election + script_df.congress +  script_df['right-wing'] + script_df.immigration +  script_df.environment + script_df.campaign + script_df.clinton +  script_df.obama + script_df.cheney
    script_df.drop(labels = ['president', 'political', 'election', 'congress', 'right-wing', 'immigration', 'environment', 'campaign', 'clinton', 'obama', 'cheney'], axis = 1, inplace = True)
    script_df = script_df[['episode_id', 'politics']].groupby(by=['episode_id'], as_index=False).sum()
    script_df.fillna(0, inplace=True)
    print(script_df.shape)
    merged_df = episode_df.merge(script_df, how='left', left_on = 'id', right_on='episode_id')
    print(merged_df.shape)
    return merged_df

def initialize(char, char_weight, location, loc_weight, song, politics):
    print('hi')
    episode_df, scripts = filter_df(char)
    episode_df = filter_locations(episode_df, location, scripts)
    episode_df.drop(labels=['production_code', 'us_viewers_in_millions', 'imdb_votes', 'episode_id_x', 'char_lines', 'non_char_lines', 'episode_id_y', 'loc_lines', 'non_loc_lines', 'views'], axis=1, inplace=True)
    print('hello')
    episode_df = get_song(episode_df, scripts, song)
    episode_df = politicize(episode_df, scripts, politics)
    if char_weight > loc_weight:
        x = episode_df.sort_values(by=['imdb_rating', 'song', "politics", "char_ratio", 'loc_ratio'], ascending = False)
        print(x.head(2))
        return x
    else:
        x = episode_df.sort_values(by=['imdb_rating', 'song', "politics", "loc_ratio", 'char_ratio'], ascending = False)
        print(x.head(2))
        return x


def return_suggested(pred_list):
    char = pred_list[0]
    char_weight = pred_list[1]
    location = pred_list[2]
    loc_weight = pred_list[3]
    song = pred_list[4]
    politics = pred_list[5]
    recommended_id = initialize(char, char_weight, location, loc_weight, song, politics)
    # print(recommended_id)
    df = ld.load_data()
    return df[df['id']==recommended_id][['title', 'predicted', 'imdb_rating', 'image_url', 'video_url']].head(1)



if __name__ == '__main__':
    initialize(['homer', 2, 'home', 3, 0, 1])
