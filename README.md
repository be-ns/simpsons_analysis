
![The Simpson's Analysis](logo_simpsonian.png)
<img src=http://solarentertainmentcorp.com/solar_uploads/ckeditor/JackTV/The_Simpsons.jpg width=100%>

# Predicting IMDB ratings from the scripts of one of America's longest running shows.
# Building a ratio-based Recommender for finding an episode you'd like.

---
## TABLE OF CONTENTS
1. [Overview](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#overview)
 * [Goal](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#goal)
 * [Process](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#process)
 * [Results](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#results)
 * [Tools Used](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#tools-used)
2. [Technical Approach](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#technical-approach)
 * [Data](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#data)
 * [Cleaning / Munging](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#munging--cleaning)
 * [Model Selection / Benchmarking](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#model-selection--benchmarking)
 * [Modeling Process](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#modeling--algorithms)
 * [Error Metric](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#error-metric-choice)
 * [Features](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#model---features)
 * [Model Rationale](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#model---rationale)
 * 
3. [Future Steps](https://github.com/be-ns/simpsons_analysis/blob/master/README.md#next-steps)

---

### OVERVIEW
#### GOAL
The purpose of this project is twofold.  
1. I engineered a model using the scripts of an animated show to accuractely predict the public response to the show.
2. I built a simple episode recommender using a ratio for preferred characters and locations as well as if the user enjoys music and politics in their animated shows.  

Ultimately, a model like this could save the creators of television shows hundreds of thousands of dollars a year. A single episode of the Simpsons costs between $400,000 and $2mm between animation, voice acting, sound, and final production. If the writers utilized a tool like this IMDB predictor, they would be able to catch potentially low ratings before going into production and rework the episode.  

#### PROCESS
##### Stacked IMDB Prediction Model
I engineered features from the 600 scripts of the Simpsons ranging from 1989 to 2016. Analysis was done on the text of the episode scripts to grab key episode data like words in the episode, character and location information, and the ratio of lines spoken by the core character group.
Using this feature matrix coupled with the IMDB ratings, I built a [stacked model](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=5&cad=rja&uact=8&ved=0ahUKEwj2wOyt28rUAhVLKiYKHfTeBM8QFghAMAQ&url=http%3A%2F%2Fblog.kaggle.com%2F2016%2F12%2F27%2Fa-kagglers-guide-to-model-stacking-in-practice%2F&usg=AFQjCNG_KRzJK5q6ajl3t7tpvJTil6YlQw&sig2=SqEKyASqCvxsmshFIHmR4g) using two Boosted Decision Trees (Adaboost and Gradient Boosting) that were optimized using a [random parameter search](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881). This model was run overnight on an EC2 instance, recursively pickling the model whenever a better hold-out error was achieved. This, combined with the random-search, was cross-validated to ensure the model was not resting at a local minimum.
###### Quick Visualization of How [Gradient Boosting](https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting) works
<img src=https://openi.nlm.nih.gov/imgs/512/70/4538588/PMC4538588_CIN2015-149702.002.png width = 100%>  

###### The `Error` is weighted each time and a step is taking in the negative direction of the gradient until the loss function is at the lowest it could be.

##### Recommender / Web App
The recommender I built is not a _true_ recommmender in the sense of using distance metrics to compare similarity. This was chosen since the data available was only the scripts for the episodes. There were no user ratings, and since most people would not know an episode well enough to ask for similar episodes, I chose to look at it with a 'cold-start' mindset. For this reason a user inputs preferences such as a favorite location or if they like songs in their animated shows. The recommender then finds and sorts every episode by the selected preferences and returns the highest rated episode that meets their criteria. Since this process would take anywhere from 20-40 seconds, I built a hash-table using Python dictionaries and stored the results for every combination of preferences it was possible to have.
The web-app shows a thumbnail of the suggested episode, along with the predicted and the actual values for the episode (using the stacked model from above), and a button to click and watch the episode.
[![screenshot of web app](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/Flask_app.png?raw=true)](http://ec2-34-203-221-235.compute-1.amazonaws.com:8080/)

#### RESULTS
The __Stacked Model__ using engineered features, random-search hyperparameter optimization and recursive attempts to outscore itself, achieved a RMSE of 0.351 on a 1-10 scale (RMSE is in the same units as the target number).  

![image](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/score.png?raw=true)

The __Recommender__ is stored on Amazon Web Services and can be played around with [here](http://ec2-34-203-221-235.compute-1.amazonaws.com:8080/)

#### TOOLS USED:
* [Python](https://www.python.org)
* [Pandas](http://pandas.pydata.org/index.html)
* [Numpy](https://docs.scipy.org/doc/numpy-1.12.0/reference/)
* [SciKitLearn](http://scikit-learn.org/stable/)
* [MatPlotLib](https://matplotlib.org/)
* [SciPy](https://www.scipy.org/)
* [Flask](http://flask.pocoo.org/)
* [AWS (S3 and EC2)](https://www.aws.amazon.com])
---
### TECHNICAL APPROACH
#### DATA
[Original Dataset](https://github.com/be-ns/simpsons_analysis/blob/master/data/simpsons_episodes.csv) was in CSV format, delineated so that each spoken line, whether 1 or 140 words long, had it's own line. It was found on [Data World](https://data.world/data-society/the-simpsons-by-the-data) in four relational csv files (grouped either by location_id, character_id, or episode_id. Features included episode ID, season, and number in series (season four episode three and season 12 episode three would both have a number in series of 3).
In a [second dataset](https://github.com/be-ns/simpsons_analysis/blob/master/data/simpsons_script_lines.csv) there was episode specific information like original air date, imdb rating, and viewership upon initial airing.

#### MUNGING / CLEANING
Initial data exploration was done in Pandas, Python, and MatPlotLib using a Jupyter Notebook.
Data was cleaned in Python using a variety of packages, most heavily Pandas and Numpy. 
Script lines needed to have 37 rows manually cleaned due to double-quotation errors from in the `text` and `raw_text` sections
`NaN`s were imputed with the column-mean. Given more time, I would like to improve this to my preferred method of K-Nearest Neighbor NaN imputation. Alternately, using a `backfill` method could be useful for timeseries data. The spoken lines (`raw_text`) was the only aspect of the script information used. I did not use screen direction or animation notes.

![distribution of scores](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/distribution_of_scores.png?raw=true)

The Data was split into a `train` and `holdout` set (80% / 20% breakdown). Models were built using the training set with the final model selected based on the best hold-out RMSE. The algorithm was run in a while loop on an EC2 instance overnight, allowing the model to overwrite any saved models when the score improved. The overnight score went from .042 to 0.351; the  model for the latter was saved (i.e. pickled). Persisting the model file allows me to skip the training step in the future.

#### MODEL SELECTION / BENCHMARKING
Data benchmarking was done with minor hyperparameter optimization for several algorithms. All models were trained on the training set (80% of original data) and scores below are for 3-fold cross-validated models for both training error and test error.  
###### Benchmarking Train/Test error for regression algorithms.

![models](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/error_rates.png?raw=true)

###### For this graphic, the pickled Stacked Model was utilized, so the RMSE below 0.3 is not accurate to the holdout score on initial model training.

The model selected was chosen due to the non-linear feature-to-target connections. Flexibility was highly valued, as was reduction in compute power. Although K-Nearest Neighbors was the quickest model, the test error showed it didn't improve to the extant that the stacked model did.

#### MODELING / ALGORITHMS
Stacking (also known as meta ensembling) is an ensemble modeling technique used to combine information from multiple predictive models to generate a new, better model. Often times the stacked model (or 2nd-level model) will outperform each of the individual models due its smoothing nature (reducing variance from overfitting) and the ability to highlight each base model where it performs best and discredit each base model where it performs poorly.

<center>
<img src=https://cdn1.tnwcdn.com/wp-content/blogs.dir/1/files/2016/02/Screen-Shot-2016-02-03-at-15.59.14.png width=50%></center>

For my stacked model I chose to use AdaBoost for the initial model, and Gradient Boosting for the 2nd-level model.  
Both of these are sequentially built Decision trees that aim to minimize a loss function by weighting incorrect predictions and then rebuilding the decision tree. Gradient Boosting does so by taking a step in the negative direction of the gradient (which is a fancy way of saying it tries to minimize the error to zero by going back and rebuilding itself while emphasizing certain data points). 
This method of stacking two models together resulted in a RMSE of 0.351, meaning my predicted IMDB ratings were off on average by .351 on a 0 to 10 scale.

#### ERROR METRIC CHOICE
I chose to evaluate my model using the Square Root of the Mean Squared Error, or RMSE. RMSE is calculated by finding the average of all the squared errors, then taking the square root. The resulting metric is a positive average error in the original metric of the equation (in this case, it is the amount my 1-10 scale prediction is off by).  

<img src=http://file.scirp.org/Html/htmlimages/5-2601289x/fcdba7fc-a40e-4019-9e95-aca3dc2db149.png width=50%>

###### Such that `N` = number of predictions made; Y-hat = Predicted score; Y = true score

#### MODEL - FEATURES
The final Stacked Model used engineered features to capture the signal for the Simpsons IMDB ratings. The top features in the model were the length of the lines (number of words in the average statement), the longest line in the episode, and the `Simpson to Other` Ratio. This Ratio was found by analyzing the percent of lines in the episode spoken by the core group of characters. A larger `Simpson to Other` Ratio implies that a large percentage of the episode was spoken lines by the Simpson family. Another feature of note was the Political Cycle boolean, which was derived from analyzing if the episode aired during a political cycle or not. The Simpson's is often explicitly political in nature, which I thoungt may or may not influence the public response to the episode. 
Note the non-linear signal of the data, which lended itself nicely to the 
![feature one](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/avg_line_len.png?raw=true)
![feature two](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/Simpsons_to_other.png?raw=true)
![feature three](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/longest_line.png?raw=true)

#### MODEL - RATIONALE
##### Why not NLP?
1. At this stage, the data isn't consumer facing - it would be a raw script - utilizing engineered features would be easier to interpret than doing TF-IDF with PCA and more flexible than LDA. Engineered features would better serve the purpose here, since I could pull the top feaures from the stacked model.
![top features](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/feature_importances.png?raw=true)
##### Why not Parallelize?
1. The algorithms I utilized are built sequentially, requiring previous knowledge to be built correctly. Parallelizing the data here would be impossible with this model. 
2. The models were small enough to fit in memory, allowing lower latency than spinning up clusters in the cloud.

---
### NEXT STEPS
1. Impute NaNs with KNN instead of column-mean.
2. Experiment with Polynomial Expansion for feature space to see if this would improve accuracy.
3. Alter political cycle metric to inlcude counts of political language.
4. Give actionble insights from model for altering scripts to increase publice rating.
5. Allow model to return top-N episodes.
6. Run videos natively on Flask App.
7. Pull Wikipedia description for every episode to be displayed on page.
8. Compare stacked model with PyMC2 model.


<img src=https://cdn1.vox-cdn.com/uploads/chorus_asset/file/4204155/simp_Cue_Detective_TABF17_T17_Sc1074_hires2.0.jpg width=75%>

---

Special Thanks to [Todd Schneider](http://toddwschneider.com/posts/the-simpsons-by-the-data/) for his Simpsons by the Data analysis, which inspired the project
