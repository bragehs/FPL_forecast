
## FOLDERS OVERVIEW
code is generally very messy right now. plan to make it better
### predictor
python code to train an LSTM and using pulp to get the best team for each gameweek. 

#### preprocessing.py
contains data preprocessing. in player_lag_features, I am lagging the features up to, and not including, GW n. Then in create_sequences the target value becomes total points in GW n. 

#### run_simulation.ipynb
loads preprocessed data and trained model and runs the FPL performance simulations.

#### model.py
model architectures. only using AdvancedLSTM currently.

#### eval.py
using LP to get the best team for a GW or suggested transfer.  

### src
ignore this currently. contains code from a previous project which will not be used now. i am just using the same repo to avoid setting everything up again, since the previous project was not anything good. 

## DONE

predictor folder contains scripts which, from a csv file fetched from https://github.com/vaastav/Fantasy-Premier-League, produces an LSTM model that simulates a score in FPL. There are 2 different scores. The first score is 2055 and would be a "legal" FPL performance. This score is found by selecting an initial team based on last season and then doing some transfers. Getting a good automated strategy is very demanding though. Accounting for risk of transfers, incorporating wildcards and triple captain etc. Therefore i decided to let the model showcase its raw predictive power by letting it pick its dream team for each gameweek, still within constraints. Basically if you had 38 free hit cards. Then the model produced a score of 2356, which is better than i personally got last year. It is obviously a huge difference to be able to change the entire team, but it suggests the model is performing decently. There is no triple captain or bench boost included though. 

This model is probably best used as a tool for a human to actually play FPL, instead of it playing FPL for you. I want to display predicted points for next gameweek for all relevant players on the web, and also show "dream team" which can be used with a free hit or wildcard. 
## TODO

use the FPL public API to fetch data for the 25/26 season and use the model to get predictions for next gameweek for all players. Also show the dream team for the next gameweek (within the squad constraints). 

how should the model´s output get to the web?
	- current thinking: deploy the model on the web and then call it on load or interaction. this way i do not have to implement my own backend properly. 

position should be onehot encoded not label encoded

