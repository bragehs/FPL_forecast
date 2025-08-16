import pulp
import pandas as pd
import numpy as np
import os
import torch

data_path = os.getcwd() + '/processed_data'

def make_available_players_df(this_season_player_df, last_season_player_df):
    
    last_season_player_df = last_season_player_df[last_season_player_df.minutes > 0]
    last_season_player_df = last_season_player_df[['name', "total_points"]]
    last_season_player_df.rename(columns={'total_points': "total_points_last_season"},
                                inplace=True)
    
    available_players_df = pd.merge(this_season_player_df,
                                    last_season_player_df,
                                   on='name', how='left')

    # First attempt: fill by position and value groups
    available_players_df['total_points_last_season'] = available_players_df.groupby(['position_encoded', 'value'])['total_points_last_season'].transform(lambda x: x.fillna(x.mean()))
    
    # Second attempt: fill by position only if still NaN
    available_players_df['total_points_last_season'] = available_players_df.groupby(['position_encoded'])['total_points_last_season'].transform(lambda x: x.fillna(x.mean()))
    
    nan_values = available_players_df[available_players_df['total_points_last_season'].isna()]
    print("Players with NaN total_points_last_season:", nan_values['name'].unique())
    print("Number of NaN values remaining:", len(nan_values))
    
    return available_players_df

def get_cheapest_players(player_df):
    
    cheapest_player_names = []
    total_cost = 0
    
    # for each position, sort the players by cost (in ascending order)
    # then, get the player with the most number of points
    
    for position, group in player_df.groupby('position_encoded'):
        cheapest_players =  group[(group.value == group.value.min())]
        top_cheapest_player = cheapest_players[cheapest_players['total_points'] == cheapest_players['total_points'].max()]

        cheapest_player_name = top_cheapest_player['name'].values[0]
        
        cheapest_player_names += [cheapest_player_name]
        total_cost += top_cheapest_player.value.values[0]
        
        print(position, ": ", cheapest_player_name )
        
    return cheapest_player_names, total_cost


def get_initial_team(prob, player_df):
    variable_names = [v.name for v in prob.variables()]
    variable_values = [v.varValue for v in prob.variables()]

    # Create the decision variables DataFrame
    decision_df = pd.DataFrame({"name": variable_names, "selected": variable_values})

    # Perform merge
    initial_team = pd.merge(decision_df, player_df, on="name", how='left')
    initial_team = initial_team[initial_team["selected"] == 1.0]

    return initial_team


def make_predicted_table(y_test, y_pred):
    '''
    Create a DataFrame for LSTM model predictions.
    This needs to keep track of the Gameweek (GW) and player names.
    '''
    test_mapping = pd.read_csv(data_path + '/test_mapping.csv')
    predictions_df = test_mapping.copy()
    predictions_df['actual'] = y_test
    predictions_df['predicted'] = y_pred
    predictions_df['predicted'] = predictions_df['predicted'].where(predictions_df['minutes'] > 0, 0)
    predictions_df.rename(columns={'prediction_gw': 'GW'}, inplace=True)


    # Update the predictions_df reference
    predictions_df = predictions_df.drop_duplicates(subset=['name', 'GW'], keep='last')
    
    return predictions_df


def get_score(team_list, gw_df, sort_by='predicted'):
    gw_score = gw_df[gw_df['name'].isin(team_list)].actual.sum() \
        + gw_df[(gw_df['name'].isin(team_list)) & (gw_df['position_encoded'].isin([3, 4]))].sort_values(sort_by, ascending=False).head(1).actual.values[0] #only captain midfielders or strikers

    print(gw_df[gw_df['name'].isin(team_list)][['name', 'actual', 'predicted']])
    print("total_score for gameweek", gw_df['GW'].values[0], ":", gw_score)
    return gw_score


def season_performance_with_unlimited_transfers(y_test, predictions, remaining_lagged_features):
    """
    Get the season performance of the team based on the predictions with unlimited transfers.
    Creates a completely new optimal team for each gameweek using that gameweek's predictions.
    """
    previous_season = pd.read_csv(data_path + '/val_data.csv')
    test = pd.read_csv(data_path + '/test_data.csv')

    group_key = 'name'
    first_cols = ['team_x','season_x', 'position_encoded', 'element', 'value', 'position', 'was_home']

    # Only use non-grouping columns in aggregation
    agg_dict = {
        col: 'first' if col in first_cols else 'sum'
        for col in test.columns
        if col != group_key
    }

    summed_test = test.groupby('name', as_index=False).agg(agg_dict)
    summed_last_season = previous_season.groupby('name', as_index=False).agg(agg_dict)
    summed_last_season['value'] = summed_last_season['value'].astype(float)
    summed_test['value'] = summed_test['value'].astype(float)

    # Create the predicted table for all gameweeks
    predicted_df = make_predicted_table(y_test, predictions)
    
    # Get available players (merge current season with previous season data)
    available_players_df = make_available_players_df(summed_test, summed_last_season)
    
    # Get bench players and cost (needed for budget calculation)
    bench_player_names, bench_cost = get_cheapest_players(available_players_df)
    print("Bench players:", bench_player_names)
    print("Bench cost:", bench_cost)
    
    # Available budget for main team (1000 - bench cost)
    available_budget = 1000 - bench_cost
    
    gameweeks = sorted(test['GW'].unique())
    total_score = 0
    gw_scores = []
    gw_teams = []
    
    print(f"Simulating season with unlimited transfers for {len(gameweeks)} gameweeks")
    print(f"Available budget per gameweek: {available_budget}")
    
    for gw in gameweeks:
        print(f"\n--- Gameweek {gw} ---")
        
        # Get predictions for this specific gameweek
        gw_predictions = predicted_df[predicted_df['GW'] == gw].copy()

        # Create a temporary available players dataframe with this gameweek's predictions
        # Merge current gameweek predictions with player info
        gw_player_data = pd.merge(
            available_players_df[['name', 'team_x', 'position_encoded', 'value', 'total_points_last_season']],
            gw_predictions[['name', 'predicted']],
            on='name',
            how='inner'
        )
        
        # Only consider players who are predicted to play (have predictions)
        gw_player_data = gw_player_data.dropna(subset=['predicted'])
        
        print(f"Players available for GW {gw}: {len(gw_player_data)}")
        
        if len(gw_player_data) == 0:
            prob = solve_optimization_problem_for_gameweek(available_players_df, bench_cost, gw, maximizing_variable="total_points_last_season")
            initial_team_df = get_initial_team(prob, available_players_df)
            my_team = initial_team_df['name'].tolist()
            print("My team:", my_team)
            gw_score = get_score(my_team, gw_predictions, sort_by='actual')
            total_score += gw_score
            gw_teams.append(my_team)
            gw_scores.append(gw_score)
            continue
        
        # Solve optimization problem for this gameweek using predicted points
        try:
            prob = solve_optimization_problem_for_gameweek(gw_player_data, bench_cost, gw)
            
            if prob.status == 1:  # Optimal solution found
                # Get the optimal team for this gameweek
                optimal_team_df = get_initial_team(prob, gw_player_data)
                optimal_team_names = optimal_team_df['name'].tolist()
                
                # Calculate score for this gameweek using actual points
                gw_score = get_score(optimal_team_names, gw_predictions)
                
                total_score += gw_score
                gw_scores.append(gw_score)
                gw_teams.append(optimal_team_names)
                
                print(f"GW {gw} optimal team: {optimal_team_names}")
                print(f"GW {gw} score: {gw_score}")
                print(f"Team cost: {optimal_team_df['value'].sum()}")
                
            else:
                print(f"Could not find optimal solution for GW {gw}")
                gw_scores.append(0)
                gw_teams.append([])
                
        except Exception as e:
            print(f"Error solving optimization for GW {gw}: {e}")
            gw_scores.append(0)
            gw_teams.append([])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'team': gw_teams,
        'gw_score': gw_scores
    })
    
    print(f"\n=== UNLIMITED TRANSFERS SEASON SUMMARY ===")
    print(f"Total season score: {total_score}")
    print(f"Average GW score: {np.mean(gw_scores):.2f}")
    print(f"Best GW score: {max(gw_scores) if gw_scores else 0}")
    print(f"Worst GW score: {min(gw_scores) if gw_scores else 0}")
    
    return results_df, total_score


def solve_optimization_problem_for_gameweek(gw_player_data, bench_cost, gw_num, maximizing_variable="predicted"):
    """
    Solve optimization problem for a specific gameweek using predicted points as the objective.
    """
    available_cash = 1000 - bench_cost
    
    prob = pulp.LpProblem(f'OptimalTeam_GW{gw_num}', pulp.LpMaximize)
    
    # Create decision variables
    decision_variables = [pulp.LpVariable(name, cat="Binary") for name in gw_player_data['name']]
    
    # Objective function: maximize predicted points for this gameweek
    objective = ""
    for i, player in enumerate(decision_variables):
        objective += gw_player_data.iloc[i][maximizing_variable] * player
    prob += objective
    
    # Budget constraint
    budget_constraint = ""
    for i, player in enumerate(decision_variables):
        budget_constraint += gw_player_data.iloc[i]['value'] * player
    prob += (budget_constraint <= available_cash)
    
    # Position constraints (1 GK, 4 DEF, 4 MID, 2 FWD)
    for position in [1, 2, 3, 4]:  # Changed from [0, 1, 2, 3] to [1, 2, 3, 4]
        position_constraint = ""
        required_count = [1, 4, 4, 2][position-1]  # Adjusted index: GK, DEF, MID, FWD
        
        for i, player in enumerate(decision_variables):
            if gw_player_data.iloc[i]['position_encoded'] == position:
                position_constraint += 1 * player
        
        prob += (position_constraint == required_count)
    
    # Team constraint (max 3 players per team)
    for team in gw_player_data['team_x'].unique():
        team_constraint = ""
        team_players = gw_player_data[gw_player_data['team_x'] == team]
        
        for i, player in enumerate(decision_variables):
            if gw_player_data.iloc[i]['name'] in team_players['name'].values:
                team_constraint += 1 * player
        
        prob += (team_constraint <= 3)
    
    # Solve the problem
    prob.solve()  # Silent solver
    
    return prob