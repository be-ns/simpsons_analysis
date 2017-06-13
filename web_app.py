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
    simpsons_characters = pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_characters.csv')


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
@app.route('/go', methods=['GET', 'POST'])
def get_activity_predictors():
    char_list = list(pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_characters.csv').index)
    char_weight = [1,2,3,4,5]
    fav_location = list(pd.read_csv('/Users/benjamin/Desktop/DSI/simpsons_analysis/data/simpsons_locations.csv').index)
    loc_weight = [1,2,3,4,5]
    song = [False,True]
    politics = [False,True]
    return render_template('form.html', char_list=char_list, char_weight = char_weight, fav_location = fav_location, loc_weight=loc_weight, song = song, politics = politics)

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

    # list contains:
    #       ['title', 'predicted', 'imdb_rating', 'image_url', 'video_url']
    episode_list = fe.return_suggested(pred_list)
    return render_template('results.html', data=pred_list)

app.secret_key = os.urandom(24)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9002, debug=False)
