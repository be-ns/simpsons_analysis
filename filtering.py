import pandas as pd
import simpsons as s
import numpy as np
from string import punctuation


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

    # group all script lines by episode and get counts of character speaking lines (divided by character)
    script_df = script_df.groupby(['episode_id', 'raw_character_text'], as_index=False)['speaking_line'].count()

    # find the chosen characters lines
    char_lines = script_df[script_df['raw_character_text'].str.contains(char)] \
            .groupby('episode_id', as_index = False)['speaking_line'].sum()
    # find all other character lines summed lines
    otherlines =script_df[~script_df['raw_character_text'].str.contains(char)] \
            .groupby('episode_id', as_index = False)['speaking_line'].sum()
    # merge the two into a single DF
    char_ratio = char_lines.merge(otherlines, how = 'left', left_on = 'episode_id', right_on='episode_id')
    # reset names
    char_ratio.columns = ['episode_id', 'char_ratio', 'otherlines']
    # make char_ratio a ratio and drop the otherlines
    char_ratio['char_ratio'] = round((char_ratio['char_ratio'] / char_ratio['otherlines']), 3)
    char_ratio.drop(['otherlines'], axis=1, inplace=True)

    # merge that into the episode DF and return DF
    merged_df = episode_df.merge(char_ratio, how='left', left_on = 'id', right_on='episode_id')
    merged_df.char_ratio.fillna(0, inplace=True)
    # print(merged_df.head())
    return merged_df, scripts

def filter_locations(episode_df, location, scripts):
    script_df = pd.read_csv(scripts)
    script_df['raw_location_text'] = np.array([str(x).lower() for x in script_df.raw_location_text])
    script_df = script_df.groupby(['episode_id', 'raw_location_text'], as_index=False).spoken_words.count()
    script_df.columns = ['id', 'location', 'lines_in_locations']

    # find the chosen location lines
    loc_lines = script_df[script_df['location'].str.contains(location)] \
            .groupby('id', as_index = False)['lines_in_locations'].sum()
    # find all other character lines summed lines
    othrlines = script_df[~script_df['location'].str.contains(location)] \
            .groupby('id', as_index = False)['lines_in_locations'].sum()

    loc_ratio = loc_lines.merge(othrlines, how = 'left', left_on = 'id', right_on='id')
    # reset names
    loc_ratio.columns = ['episode_id', 'loc_ratio', 'otherlines']
    # make char_ratio a ratio and drop the otherlines
    loc_ratio['loc_ratio'] = round((loc_ratio['loc_ratio'] / loc_ratio['otherlines']), 3)
    loc_ratio.drop(['otherlines'], axis=1, inplace=True)

    # merge that into the episode DF and return DF
    merged_df = episode_df.merge(loc_ratio, how='left', left_on = 'id', right_on='episode_id')
    merged_df.loc_ratio.fillna(0, inplace=True)
    return merged_df

def get_song(episode_df, s_b, scripts):
    script_df = pd.read_csv(scripts)
    script_df['song'] = script_df['raw_text'].apply(lambda x: 1 if 'singing' in str(x).lower() else 1 if 'sing' in str(x).lower() else 0)
    # `sie` to stand for singers in episode
    script_df['sie'] = script_df['raw_character_text'].apply(lambda x: 1 if 'singer' in str(x) else 0)
    script_df['song'] = script_df['song'] + script_df['sie']
    script_df.drop('sie', axis = 1, inplace = True)
    if s_b:
        script_df = script_df[['episode_id', 'song']][script_df['song'] != 0]
    else:
        script_df = script_df[['episode_id', 'song']][script_df['song'] == 0]
    merged_df = episode_df.merge(script_df, how='left', left_on = 'id', right_on='episode_id')
    merged_df.song.fillna(0.0, inplace=True)
    return merged_df[merged_df['song'] >= 1].sort_values('episode_id', ascending=True)

def politicize(episode_df, scripts, p_b):
    script_df = pd.read_csv(scripts)
    politics = 'president politics political election congress right-wing immigration environment campaign clinton obama'.lower().split(' ')
    if p_b:
        for word in politics:
            script_df[word] = script_df['raw_text'].apply(lambda x: 1 if word in str(x).lower() else 0)
        script_df['politics'] = script_df.president + script_df.politics +  script_df.political +  script_df.election + script_df.congress +  script_df['right-wing'] + script_df.immigration +  script_df.environment + script_df.campaign + script_df.clinton +  script_df.obama
        script_df.drop(labels = ['president', 'political', 'election', 'congress', 'right-wing', 'immigration', 'environment', 'campaign', 'clinton', 'obama'], axis = 1, inplace = True)
        script_df = script_df[script_df.politics != 0.0][['episode_id', 'politics']]
    else:
        for word in politics:
            script_df[word] = script_df['raw_text'].apply(lambda x: 1 if word in str(x).lower() else 0)
        script_df['politics'] = script_df.president + script_df.politics +  script_df.political +  script_df.election + script_df.congress +  script_df['right-wing'] + script_df.immigration +  script_df.environment + script_df.campaign + script_df.clinton +  script_df.obama
        script_df.drop(labels = ['president', 'political', 'election', 'congress', 'right-wing', 'immigration', 'environment', 'campaign', 'clinton', 'obama'], axis = 1, inplace = True)
        script_df = script_df[script_df.politics == 0.0][['episode_id', 'politics']]
    merged_df = episode_df.merge(script_df, how='left', left_on = 'id', right_on='episode_id')
    return merged_df

def initialize():
    #set dict for future weighting of episodes to be returned
    weights = {'char':0, 'location':0}

    # filter on Character
    char = input('favorite Simpsons Character  :  ').lower().strip().strip(punctuation)
    #update weights
    weights['char'] = int(input('How important is it that your suggested episode features this character [1-10]  :  ').strip())
    # call function to get character ratios
    episode_df_, scripts = filter_df(char)

    # filter on Location
    location = input('favorite Simpsons Location - type [H] to see options : ').lower().strip().strip(punctuation)
    # update weights
    weights['location'] = int(input('How important is it that your suggested episode features this location [1-10]  :  ').strip())
    # don't let location be nothing
    while location == 'h':
        loc_df = pd.read_csv('data/simpsons_locations.csv')
        print([x.lower().strip(punctuation) for x in loc_df.as_matrix()[:,1]])
        location = input('favorite Simpsons Location  :  ').lower().strip().strip(punctuation)
    # once they have a location selected
    episode_df = filter_locations(episode_df_, location, scripts)
    episode_df.drop(labels=['imdb_votes', 'episode_id_x', 'episode_id_y'], axis=1, inplace=True)

    # filter on if songs appear in the episode
    song_bool = input("Do you like songs in your Simpson's Episodes? [N], [y] : ").strip(punctuation).lower()[0].strip()
    episode_df = get_song(episode_df, song_bool, scripts)

    # filter on political episode or not
    political_bool = input("Do you enjoy political humor or topics within your Simpsons episode? [Y], [n]  :  ").strip(punctuation).lower()[0].strip()
    episode_df = politicize(episode_df, scripts, political_bool)

    # return the episode name and the link to view it
    if weights['char'] > weights ['location']:
        print(episode_df.sort_values(by=['char_ratio', 'loc_ratio', 'song', 'politics', 'imdb_rating'], ascending=False)  \
                .head(1)[['id','title']])
    else:
        print(episode_df.sort_values(by=['loc_ratio', 'char_ratio', 'song', 'politics', 'imdb_rating'], ascending=False)  \
                .head(1)[['id','title']])

if __name__ == '__main__':
    initialize()
