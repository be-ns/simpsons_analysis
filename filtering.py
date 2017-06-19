import pandas as pd
import simpsons as s
import numpy as np
from string import punctuation
import loading_data as ld


def filter_df(
            char, scripts="data/simpsons_script_lines.csv",
            episodes="data/simpsons_episodes.csv"
            ):
    '''
    INPUT: filepaths for script data and episode data (defaulted)
    OUTPUT: Pandas DataFrame with ratio of speaking
    lines for user inputted character for every episode

    NaNs are filled after ratio for minor characters
    so any episodes without the character included would be eliminated
    '''
    # read in filepaths
    script_df = pd.read_csv(scripts)
    episode_df = pd.read_csv(episodes)
    # ensure all character names are lowercased
    script_df['raw_character_text'] = np.array(
                                        [str(x).lower() for x in script_df.raw_character_text])
    # find the chosen characters lines
    char_series = script_df['raw_character_text'] \
        .apply(
            lambda x: 1 if char.lower().strip() in str(x) else 0
            )
    script_df['char_lines'] = char_series

    # find all other character lines summed lines
    otherlines = script_df['raw_character_text'] \
        .apply(
            lambda x: 1 if not char.lower().strip() in str(x) else 0
            )
    script_df['non_char_lines'] = otherlines
    script_df = script_df.groupby(
                            'episode_id', as_index=False
                            )[['char_lines', "non_char_lines"]].sum()
    # make char_ratio
    script_df['char_ratio'] = (
        script_df['char_lines'] / script_df['non_char_lines']
        )
    # merge that into the episode DF and return it
    return episode_df.merge(
                        script_df, how='left', left_on='id',
                        right_on='episode_id'
                        ), scripts


def filter_locations(episode_df, location, scripts):
    '''
    INPUT: DF of episode information, the filepath of
    the location csv, the filepath of the script csv
    OUTPUT: merged DF (filtered)
    '''
    script_df = pd.read_csv(scripts)
    script_df['raw_location_text'] = np.array(
        [str(x).lower() for x in script_df.raw_location_text]
        )

    loc_series = script_df['raw_location_text'].apply(
        lambda x: 1 if location.lower().strip() in str(x) else 0
        )
    script_df['loc_lines'] = loc_series

    # find all other location lines summed lines
    otherlines = script_df['raw_location_text'] \
        .apply(lambda x: 1 if not location.lower().strip() in str(x) else 0)
    script_df['non_loc_lines'] = otherlines
    script_df = script_df.groupby(
        'episode_id', as_index=False
        )[['loc_lines', "non_loc_lines"]].sum()
    # make loc_ratio
    script_df['loc_ratio'] = (
        script_df['loc_lines'] / script_df['non_loc_lines']
        )
    # merge that into the episode DF with an INNER JOIN and return it
    return episode_df.merge(
                        script_df, how='left',
                        left_on='id', right_on='episode_id'
                        )


def get_song(episode_df, scripts, song):
    '''
    INPUT: DF of episode information, filepath for the script csv,
    a Boolean for filtering on song or on `NOT` song
    OUTPUT: merged_df (filtered)
    '''
    # read in information
    script_df = pd.read_csv(scripts)
    # Boolean contingent
    if song:
        script_df['song'] = script_df['raw_text'] \
            .apply(lambda x: 1 if (
                'singing' in str(x).lower() or 'sing' in str(x).lower()
                ) else 0)
        # `sie` to stand for singers in episode
        script_df['sie'] = script_df['raw_character_text'] \
            .apply(lambda x: 1 if 'singer' in str(x).lower() else 0)
    elif not song:
        script_df['song'] = script_df['raw_text'] \
            .apply(lambda x: 0 if (
                'singing' in str(x).lower() or 'sing' in str(x).lower()
                ) else 1)
        # `sie` to stand for singers in episode
        script_df['sie'] = script_df['raw_character_text'] \
            .apply(lambda x: 0 if 'singer' in str(x).lower() else 1)
    # combine the two Series'
    script_df['song'] = script_df['song'] + script_df['sie']
    # drop the unneccesary one
    script_df.drop('sie', axis=1, inplace=True)
    # group the DF by episode ID and get musical score
    script_df = script_df.groupby("episode_id", as_index=False)['song'].sum()
    # return merged df
    return episode_df.merge(
                        script_df, how='left',
                        left_on='id', right_on='episode_id'
                        )


def politicize(episode_df, scripts, politics):
    '''
    INPUT: DF of episode information, filepath for scripts csv,
    Boolean value for political inclination
    OUTPUT: merged_df (filtered)
    '''
    # read in script csv
    script_df = pd.read_csv(scripts)
    # make words to be looking for
    politics_lst = 'president politics political election \
                    congress right-wing immigration environment \
                    campaign clinton obama cheney'.lower().split(' ')
    # get counts for how many times these words appear
    # in the episode / or don't
    for word in politics_lst:
        if politics:
            for word in politics_lst:
                script_df[word] = script_df['raw_text'].apply(
                                    lambda x: 1 if word in str(x).lower()
                                    else 0
                                    )
        elif not politics:
            # using a reverse ratio (how 'NOT' political is an episode.
            for word in politics_lst:
                script_df[word] = script_df['raw_text'] \
                    .apply(
                        lambda x: 0 if word in str(x).lower() else 1
                        )
    # combine all the different words scores
    script_df['politics'] = script_df.president + script_df.politics \
        + script_df.political + script_df.election + script_df.congress \
        + script_df['right-wing'] + script_df.immigration \
        + script_df.environment + script_df.campaign + script_df.clinton \
        + script_df.obama + script_df.cheney
    # drop the words since they are combined now
    script_df.drop(labels=[
                    'president', 'political', 'election',
                    'congress', 'right-wing', 'immigration',
                    'environment', 'campaign', 'clinton',
                    'obama', 'cheney'
                    ], axis=1, inplace=True)
    # groupby episode ID and get the sum of all political talk
    script_df = script_df[
        ['episode_id', 'politics']
        ].groupby(by=['episode_id'], as_index=False).sum()
    # make sure nothing gets left, but since the score is 0
    # it means we would skip that episode for the recommender
    script_df.fillna(0, inplace=True)
    # return the merged DF using INNER JOIN
    return episode_df.merge(
                        script_df, how='left',
                        left_on='id', right_on='episode_id'
                        )


def initialize(char, location, val, song, politics):
    '''
    INPUT: the list of options for the recommender to do some work on
    - needs a character, a location, a choice between
    character and location, and two Booleans
    for musical preference and political preference
    OUTPUT: Top five episodes based on inputted preferences
    '''
    # filter the episode for the selected character
    episode_df, scripts = filter_df(char)
    # filter on location
    episode_df = filter_locations(episode_df, location, scripts)
    # drop everything else
    episode_df.drop(labels=[
                    'production_code', 'us_viewers_in_millions',
                    'imdb_votes', 'episode_id_x', 'char_lines',
                    'non_char_lines', 'episode_id_y', 'loc_lines',
                    'non_loc_lines', 'views'
                    ], axis=1, inplace=True)
    # filter on song bool
    episode_df = get_song(episode_df, scripts, song)
    # filter on politics bool
    episode_df = politicize(episode_df, scripts, politics)
    # drop any NaNs
    episode_df = episode_df.dropna()
    # start the selection process with whichever
    # value was rated as more important
    if val == 'Characters':
        # reduce DF along the way to avoid always returning the same episode
        episode_df = episode_df.sort_values(
                                by=["char_ratio", "loc_ratio"],
                                ascending=[False, False]
                                )[:40][episode_df['loc_ratio'] != 0.0].sort_values(by=['song'], ascending=False)[: 20][episode_df['char_ratio'] != 0.0].sort_values(by = ["politics"],ascending = False)[: 5]
        return episode_df.sort_values(by = ['imdb_rating'],
            ascending = False).head(1).id
    else:
        episode_df = episode_df.sort_values(
                                by=["loc_ratio", "char_ratio"],
                                ascending=[False,False]
                                )[: 40][episode_df['loc_ratio'] != 0.0] \
                                .sort_values(
                                        by=['song'],
                                        ascending=False
                                        )[: 20][episode_df['char_ratio'] != 0.0]. \
                                        sort_values(
                                                by=["politics"],
                                                ascending=False
                                                )[: 5]
        return episode_df.sort_values(by=['imdb_rating'], ascending=False).head(1).id


def return_suggested(pred_list):
    '''
    INPUT: list of preferences for episode
    OUTPUT: suggested episode information

    Return the top episode for the preferences passed in
    '''
    # extract features
    char = pred_list[0]
    location = pred_list[1]
    val = pred_list[2]
    song = pred_list[3]
    politics = pred_list[4]
    recommended_id = initialize(
                        char, location, val, song, politics
                        ).head(1).values
    df = ld.load_data()
    for row in df.as_matrix():
        if row[0] == recommended_id:
            return [row[-1], row[8], row[10], row[11], row[12]]


if __name__ == '__main__':
    initialize(['homer', 2, 'home', 3, 0, 1])
