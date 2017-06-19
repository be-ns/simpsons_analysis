![The Simpson's Analysis](logo_simpsonian.png)
<img src=http://solarentertainmentcorp.com/solar_uploads/ckeditor/JackTV/The_Simpsons.jpg width=100%>
# Predicting IMDB ratings from the scripts of one of America's longest running shows.
# Building a ratio-based Recommender for finding an episode you'd like.
---
### OVERVIEW  
#### GOAL
The purpose of this project is twofold.  
1. I engineered a model using the scripts of an animated show to accuractely predict the public response to the show.
2. I built a simple episode recommender using a ratio for preferred characters and locations as well as if the user enjoys music and politics in their animated shows.

#### PROCESS
##### Stacked IMDB Prediction Model
I engineered features from the 600 scripts of the Simpsons ranging from 1989 to 2016. Analysis was done on the text of the episode scripts to grab key episode data like words in the episode, character and location information, and the ratio of lines spoken by the core character group.
Using this feature matrix coupled with the IMDB ratings, I built a [stacked model](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=5&cad=rja&uact=8&ved=0ahUKEwj2wOyt28rUAhVLKiYKHfTeBM8QFghAMAQ&url=http%3A%2F%2Fblog.kaggle.com%2F2016%2F12%2F27%2Fa-kagglers-guide-to-model-stacking-in-practice%2F&usg=AFQjCNG_KRzJK5q6ajl3t7tpvJTil6YlQw&sig2=SqEKyASqCvxsmshFIHmR4g) using two Boosted Decision Trees (Adaboost and Gradient Boosting) that were optimized using a [random parameter search](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881). This model was run overnight on an EC2 instance, recursively pickling the model whenever a better hold-out error was achieved.

##### Recommender / Web App
The recommender I built is not a _true_ recommmender in the sense of using distance metrics to compare similarity. This was chosen since the data available was only the scripts for the episodes. There were no user ratings, and since most people would not know an episode well enough to ask for similar episodes, I chose to look at it with a 'cold-start' mindset. For this reason a user inputs preferences such as a favorite location or if they like songs in their animated shows. The recommender then finds and sorts every episode by the selected preferences and returns the highest rated episode that meets their criteria. Since this process would take anywhere from 20-40 seconds, I built a hash-table using Python dictionaries and stored the results for every combination of preferences it was possible to have.
The web-app shows a thumbnail of the suggested episode, along with the predicted and the actual values for the episode (using the stacked model from above), and a button to click and watch the episode.

#### RESULTS
The __Stacked Model__ using engineered features, random-search hyperparameter optimization and recursive attempts to outscore itself, achieved a RMSE of 0.351 on a 1-10 scale (RMSE is in the same units as the target number).  

![image](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/score.png?raw=true)

The __Recommender__ is stored on Amazon Web Services and can be played around with [here](http://ec2-34-203-221-235.compute-1.amazonaws.com:8080/)

#### Tools Used:
* Python
* Pandas
* Numpy
* SciKitLearn
* MatPlotLib
* SciPy
* Flask
* AWS (S3 and EC2)
---
### Technical Approach
#### Algorithms Used
![different models](https://github.com/be-ns/simpsons_analysis/blob/master/graphs/error_rates.png?raw=true)
#### Techniques Used
#### Rationale
#### Validation Methods
#### 

While predicting a rating is nice, the ultimate goal of this project was to identify if there are any quantifiable ways to look at comedy. The Simpson's is a funny show, and the funniest episodes generally have the highest ratings. In many ways,  predicting an IMDB rating would be similar to predicting how funny an episode would be. If this is possible, it indicates the possibility that humor might not be as nuanced of a subject as we might think.

Analysis was primarily done in using Python and some Natural Langauge Processing  on the corpus. Data is present from December of 1989 October of 2016 with 600 episodes in total. 

Future steps could include performing a similar analysis on several other seasons of comedic animated series' such as Family Guy or Futurama for comparison, or using a comedic movie script to get to a Rotton Tomatoes score. Alternatively, this analysis only used the spoken lines from the show; it would be interesting to bring in unspoken directors notes for scene animations or _how_ a line would be spoken. 
