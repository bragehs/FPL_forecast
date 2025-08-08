## FOLDERS OVERVIEW
### predictor
python code to train an LSTM and using pulp to get the best team for each gameweek. 
### src
ignore this currently. contains code from a previous project which will not be used now. i am just using the same repo to avoid setting everything up again, since the previous project was not anything good. 

## DONE

predictor folder contains scripts which, from a csv file fetched from https://github.com/vaastav/Fantasy-Premier-League, produces an LSTM model that simulates a score of 2371 in FPL in 24/25. the notebook run_simulation showcases this. NB! this is with unlimited transfers as finding optimal transfer strategies is a lot of work. So the score is probably better than reality. The score was without triple captain and bench boost though. 


## TODO

use the FPL public API to fetch data for the 25/26 season and use the model to get predictions for next gameweek for all players. Also show the dream team for the next gameweek (within the squad constraints). 

how should the model´s output get to the web?
	- current thinking: deploy the model on the web and then call it on load or interaction. this way i do not have to implement my own backend properly. 

position should be onehot encoded not label encoded