# A SIMPSON'S ANALYSIS
# PREDICTING IMDB RATINGS FOR EVERY EPISODE OF THE SIMPSONS

---
The purpose of this project is twofold. First, I aim to write a machine learning script that can accuractely predict the publics rating of a comedic TV show using the scripts for the show. Second, I will build a simple episode recommender using episode specific information such as major characters or locations.

Natural Language Processing analysis will be done on the text of the episode scripts to identify common themes and grab key episode data like words in the episode, character and location information, and whether the episode has a song or not. This information will be pulled from a Relational Database containing four different sets of information (episode information, the script, location information, and character information). Once each episode has a complete set of data points, I will build a prediction script to try and accuractely predict the IMDB rating for each episode and compare it to the actual using the Square Root of the Mean Squared Error as my error metric. Following this, I will use this information to build a simple episode recommender.

While predicting a rating is nice, the ultimate goal of this project was to identify if there are any quantifiable ways to look at comedy. The Simpson's is a funny show, and the funniest episodes generally have the highest ratings. In many ways,  predicting an IMDB rating would be similar to predicting how funny an episode would be. If this is possible, it indicates the possibility that humor might not be as nuanced of a subject as we might think.

Analysis was primarily done in using Python and some Natural Langauge Processing (done in [SpaCY](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwihiYSd9_XTAhUW4WMKHezGCMIQFgg2MAE&url=https%3A%2F%2Fspacy.io%2F&usg=AFQjCNEGeNVbZtCmDfWQFUB4VPzRiaFspA&sig2=ox_-0rPFIFi1gJH0crccrA)) on the corpus. Data is present from __XXXXXXXXXX TO XXXXXXXXXY__ 2015, with 600 episodes.

Future steps could include performing a similar analysis on several other seasons of comedic animated series' such as Family Guy or Futurama with comparisons, or using a comedic movie script to get to a Rotton Tomatoes score.
