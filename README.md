
## FOLDERS OVERVIEW
code is generally very messy right now. plan to make it better
### predictor
python code to train an LSTM and using pulp to get the best team for each gameweek. 

#### preprocessing.py
contains data preprocessing. still looking for some "cheating" here but cant find it. in player_lag_features, I am lagging the features up to, and not including, GW n. Then in create_sequences the target value becomes total points in GW n. 
### src
ignore this currently. contains code from a previous project which will not be used now. i am just using the same repo to avoid setting everything up again, since the previous project was not anything good. 

## DONE

predictor folder contains scripts which, from a csv file fetched from https://github.com/vaastav/Fantasy-Premier-League, produces an LSTM model that simulates a score in FPL. There are 2 different scores, 1 which starts with an initial team and gets suggested transfers for each gameweek. Optimizing a strategy here is very demanding. The current strategy only got a score of 2292, which is decent but nothing worth using a machine learning for. but the model is limited in deciding here. I have not implemented wildcards, risk optimizing of transfers etc. Therefore i decided to let the model showcase its raw predictive power by letting it pick its dream team for each gameweek, still within constraints. Basically if you had 38 free hit cards. Then the model produced a score of 3260, which would win the entire FPL by a good margin. It is obviously a huge difference to be able to change the entire team, but it suggests the model is performing well. It seems to be changing a lot of the players as well. This model is probably best used as a tool for a human to actually play FPL, instead of it playing FPL for you. I want to display predicted points for next gameweek for all relevant players on the web, and also show "dream team" which can be used with a free hit or wildcard. 
## TODO

use the FPL public API to fetch data for the 25/26 season and use the model to get predictions for next gameweek for all players. Also show the dream team for the next gameweek (within the squad constraints). 

how should the model´s output get to the web?
	- current thinking: deploy the model on the web and then call it on load or interaction. this way i do not have to implement my own backend properly. 

position should be onehot encoded not label encoded

