from flask import Flask, request, render_template, session
import loading_data as ld
import filtering as fe
from scipy.spatial.distance import cdist
from sklearn.externals import joblib
import numpy as np
import pandas as pd # also temporary
import os


app = Flask(__name__, static_url_path = "", static_folder = "static")


# load data from loading_data.py
df = ld.load_data()

def get_simpsons_char_list(char):
    simpsons_characters = pd.read_csv('data/simpsons_characters.csv')


def get_activity_data(activities, df):
    c1, c2, c3, c4, act_ids = [],[],[],[],[]
    for activity_id in activities[:10]:
        activity = df[df['id'] == activity_id].values
        c1.append(activity[0, 1])
        c2.append(np.round(activity[0, 2] * 0.000621371, 1))
        c3.append(int(np.round(activity[0, 5] * 3.28084, 0)))
        c4.append(activity[0, 39])
        act_ids.append(activity_id)
    return zip(c1, c2, c3, c4, act_ids)
#
# def get_map_data(activity_id, df):
#     map_data = df.ix[df.id == activity_id, [-8, -7, -6]].values[0]
#     return {'sum_poly': map_data[0], 'lat': map_data[1], 'lng': map_data[2]}


# Home page with options to predict rides or runs
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# How it works page describing the recommender
@app.route('/how_it_works', methods=['GET'])
def how_it_works():
    return render_template('how_it_works_blog_post.html')

# Contact information page to link various social media
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

# This is the form page where users fill out whether they would like bike or run recommendations
@app.route('/form/<activity>', methods=['GET', 'POST'])
def get_activity_predictors(activity):
    char_list = list(pd.read_csv('data/simpsons_characters.csv').character.values)
    return render_template('form.html', city_list=city_list, displayed_activity=displayed_activity)

# This displays user inputs froms the form page
@app.route('/results', methods=['GET', 'POST'] )
def suggest_episode():
    fav_char = request.form['char']
    char_weight = request.form['char_weight']
    fav_location = request.form['location']
    loc_weight = request.form['loc_weight']
    song = 0 if request.form['musical_components'] == 'No' else 1
    politics = 0 if request.form['elevation_gain'] == 'No' else 1

    pred_list = [fav_char, char_weight, fav_location, loc_weight, song, politics]

    recommended_id = fe.return_suggested(pred_list)
    

        label = episode.
        return render_template('results.html', data=get_activity_data(rides, co_rides_df))

@app.route('/results/map/<activity_id>', methods=['GET', 'POST'])
def go_to_map(activity_id):
    activity = session['activity']
    if activity == 'bike':
        map_data = get_map_data(int(activity_id), co_rides_df)
        return render_template('map.html', data=map_data)
    else:
        map_data = get_map_data(int(activity_id), co_runs_df)
        return render_template('map.html', data=map_data)

app.secret_key = os.urandom(24)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
