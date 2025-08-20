
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
using LP to get the best team for a GW.

### src
ignore this currently. contains code from a previous project which will not be used now. i am just using the same repo to avoid setting everything up again, since the previous project was not anything good. 

## DONE

predictor folder contains scripts which, from a csv file fetched from https://github.com/vaastav/Fantasy-Premier-League, produces an LSTM model that simulates a score in FPL. Getting a good automated strategy is very demanding though. Accounting for risk of transfers, incorporating wildcards and triple captain etc. Therefore i decided to let the model showcase its raw predictive power by letting it pick its dream team for each gameweek, still within constraints. Basically if you had 38 free hit cards. Then the model produced a score of 2298, which is at least more than I usually get at least.  It is of course a huge difference to be able to change the entire team, but it suggests the model is performing decently. There is no triple captain or bench boost included though. 

This model is probably best used as a tool for a human to actually play FPL, instead of it playing FPL for you. It will of course usually predict Salah to get a lot of points, but it can provide more value in the players you do not know much about. 


I have also created an endpoint on huggingFace which returns a prediction for a player for the next gameweek when requesting with player id. 

This endpoint is used to display predictions for selected players on the website https://fpl-forecast.vercel.app/
