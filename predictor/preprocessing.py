import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import unidecode
import torch
import pickle
import json

position_mapping = {
    'GK': 1,
    'GKP': 1,
    'DEF': 2,
    'MID': 3,
    'FWD': 4,
}

player_features_to_lag = [
    'assists',
     'bonus',
     'creativity',
     'clean_sheets',
     'goals_conceded',
     'goals_scored',
     'ict_index',
     'influence',
     'minutes',
     'threat',
     'red_cards',
     'yellow_cards',
     'team_score',
     'opponent_score',
    ]


def fix_gameweek_labels(df):
    """
    Fix gameweek labels by sorting players by fixture within each season
    and reassigning GW numbers sequentially, but only when necessary.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: element, season_x, fixture, GW
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with corrected GW labels
    """
    df_fixed = df.copy()
    
    print("Analyzing GW data before fixing:")
    
    # Check each player's GW situation
    players_to_fix = []
    
    for (element, season), player_data in df_fixed.groupby(['element', 'season_x']):
        gw_count = len(player_data)
        unique_gws = player_data['GW'].nunique()
        
        # Check if player has more than 38 GWs or has duplicate GWs
        if gw_count > 38 or unique_gws < gw_count:
            players_to_fix.append((element, season))
            
            if gw_count > 38:
                print(f"Player {element} in {season}: has {gw_count} games (>38)")
            if unique_gws < gw_count:
                print(f"Player {element} in {season}: has duplicate GWs ({unique_gws} unique out of {gw_count})")
    
    print(f"Found {len(players_to_fix)} players that need GW fixing")
    
    # Only fix players that actually need fixing
    for element, season in players_to_fix:
        mask = (df_fixed['element'] == element) & (df_fixed['season_x'] == season)
        player_data = df_fixed[mask].copy()
        
        # Sort by fixture and reassign GW numbers
        player_data = player_data.sort_values('fixture').reset_index()
        
        # If player has more than 38 games, only keep first 38. This may be faulty but it only seems to happen in the training data.
        if len(player_data) > 38:
            print(f"Truncating player {element} in {season} from {len(player_data)} to 38 games")
            player_data = player_data.head(38)
            # Remove the extra rows from the main dataframe
            df_fixed = df_fixed[~((df_fixed['element'] == element) & 
                                 (df_fixed['season_x'] == season) & 
                                 (df_fixed.index.isin(player_data.iloc[38:]['index'].values)))]
        
        # Reassign GW numbers sequentially
        new_gws = list(range(1, len(player_data) + 1))
        
        # Update the main dataframe
        for i, (_, row) in enumerate(player_data.iterrows()):
            if i < len(new_gws):
                df_fixed.loc[row['index'], 'GW'] = new_gws[i]
    
    # Verify the fix worked
    print("\nChecking GW assignment after fix:")
    for season in df_fixed['season_x'].unique():
        season_data = df_fixed[df_fixed['season_x'] == season]
        gw_counts = season_data.groupby('element')['GW'].count()
        players_with_38_gws = (gw_counts == 38).sum()
        players_with_less_than_38 = (gw_counts < 38).sum()
        players_with_more_than_38 = (gw_counts > 38).sum()
        total_players = len(gw_counts)
        
        print(f"Season {season}: {players_with_38_gws}/{total_players} players have exactly 38 GWs")
        if players_with_less_than_38 > 0:
            print(f"  - {players_with_less_than_38} players have <38 GWs")
        if players_with_more_than_38 > 0:
            print(f"  - {players_with_more_than_38} players have >38 GWs")
        
        # Check GW range
        min_gw = season_data['GW'].min()
        max_gw = season_data['GW'].max()
        print(f"Season {season}: GW range is {min_gw}-{max_gw}")
        
        # Check for duplicates
        duplicate_count = season_data.groupby(['element', 'GW']).size().gt(1).sum()
        if duplicate_count > 0:
            print(f"  - Found {duplicate_count} duplicate GW entries")
    
    return df_fixed

def filter_data(df): 
    df.loc[df['position_encoded'].isna(), 'position_encoded'] = 0
    df.loc[df['fixture_difficulty'] == 1, 'fixture_difficulty'] = 2
    df.loc[df['fixture_difficulty'].isna(), 'fixture_difficulty'] = 0
    df = df[df['position'] != 'AM']  # Exclude managers
    df.dropna(inplace=True, subset=["team_h_score", "team_a_score", "team_score", "opponent_score"])
    df = df.sort_values(['season_x', 'GW']).reset_index(drop=True)
    return df

def add_fixture_difficulty_to_dataframe(df, backend_root):
    """
    Add fixture difficulty ratings to the main dataframe by merging with fixture files
    from each season directory.
    """
    base_path = backend_root + "/data/"
    
    # Get all seasons from the dataframe
    seasons = df['season_x'].unique()
    print(f"Found seasons in data: {sorted(seasons)}")
    
    # Store all fixture data
    all_fixtures = []
    
    for season in seasons:
        fixture_file = os.path.join(base_path, season, "fixtures.csv")
        
        if os.path.exists(fixture_file):
            try:
                season_fixtures = pd.read_csv(fixture_file)
                
                # Add season identifier
                season_fixtures['fixture_season'] = season
                
                # Select relevant columns and rename to avoid conflicts
                season_fixtures = season_fixtures[['id', 'team_h_difficulty', 'team_a_difficulty', 'fixture_season']]
                season_fixtures = season_fixtures.rename(columns={'id': 'fixture_id'})
                
                all_fixtures.append(season_fixtures)
                print(f"✓ Loaded {len(season_fixtures)} fixtures from {season}")
                
            except Exception as e:
                print(f"✗ Error loading fixtures from {season}: {e}")
        else:
            print(f"✗ Fixture file not found for {season}: {fixture_file}")
    
    if not all_fixtures:
        print("No fixture files found!")
        return df
    
    # Combine all fixture data
    combined_fixtures = pd.concat(all_fixtures, ignore_index=True)
    print(f"\nTotal fixtures loaded: {len(combined_fixtures)}")
    
    # Merge with main dataframe
    # The 'fixture' column in df corresponds to 'fixture_id' in fixtures
    df_with_difficulty = df.merge(
        combined_fixtures,
        left_on=['fixture', 'season_x'],
        right_on=['fixture_id', 'fixture_season'],
        how='left'
    )
    
    # Create player-specific difficulty rating
    # If player was home team, use team_h_difficulty, else use team_a_difficulty
    df_with_difficulty['fixture_difficulty'] = df_with_difficulty.apply(
        lambda row: row['team_h_difficulty'] if row['was_home'] else row['team_a_difficulty'],
        axis=1
    )
    
    # Drop temporary columns that were created during merge
    columns_to_drop = ['fixture_id', 'fixture_season']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_with_difficulty.columns]
    
    if existing_columns_to_drop:
        df_with_difficulty = df_with_difficulty.drop(existing_columns_to_drop, axis=1)
    
    print(f"\nMerge results:")
    print(f"Original dataframe: {len(df)} rows")
    print(f"With difficulty: {len(df_with_difficulty)} rows")
    print(f"Missing difficulty values: {df_with_difficulty['fixture_difficulty'].isna().sum()}")
    
    return df_with_difficulty

def add_new_features(df):
    df["team_score"] = df.apply(lambda row: row["team_h_score"] if row["was_home"] else row["team_a_score"], axis=1)
    df["opponent_score"] = df.apply(lambda row: row["team_a_score"] if row["was_home"] else row["team_h_score"], axis=1)
    df['season_progress'] = df['GW'] / df['GW'].max()
    return df

def add_future_lagged_features(df, lagged_features=['was_home', 'fixture_difficulty']):
    # Sort by player and gameweek to ensure proper ordering
    df = df.sort_values(['season_x', 'GW']).reset_index(drop=True)
    
    for feature in lagged_features:
        # Create lagged feature
        df[f'lagged_{feature}'] = df.groupby(['element', 'season_x'])[feature].shift(1)
        
        # Fill NaN values with appropriate defaults - use proper pandas method
        if feature == 'was_home':
            # For boolean, use False as default (or the current value)
            df[f'lagged_{feature}'] = df[f'lagged_{feature}'].fillna(df[feature]).infer_objects(copy=False)
        elif feature == 'fixture_difficulty':
            # Use average difficulty or current value
            df[f'lagged_{feature}'] = df[f'lagged_{feature}'].fillna(df[feature]).infer_objects(copy=False)
    
    return df

def normalize_and_encode(df, 
                     min_max_features, 
                     categorical_features, 
                     metadata_features,
                     binary_features, target,
                     scalers=None, encoders=None, fit=False,):
    """
    Preprocess the dataframe by scaling continuous features and encoding categorical features.
    
    Args:
        df: DataFrame to process
        scalers: Dictionary of fitted scalers (if fit=False)
        encoders: Dictionary of fitted encoders (if fit=False)
        fit: Whether to fit new scalers/encoders or use existing ones
    
    Returns:
        X_processed: Processed DataFrame
        scalers: Dictionary of scalers (if fit=True)
        encoders: Dictionary of encoders (if fit=True)
    """
    # Separate features
    X_min_max = df[min_max_features]
    X_categorical = df[categorical_features]
    X_metadata = df[metadata_features]
    X_binary = df[binary_features]

    if fit:
        # Fit new scalers and encoders
        scalers = {}
        encoders = {}
        
        # Scale continuous features
        scalers['continuous'] = MinMaxScaler()
        X_minmax_scaled = scalers['continuous'].fit_transform(X_min_max)
        
        # Encode categorical features
        encoders['categorical'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical_encoded = encoders['categorical'].fit_transform(X_categorical)
        
    else:
        # Use existing scalers and encoders
        X_minmax_scaled = scalers['continuous'].transform(X_min_max)
        X_categorical_encoded = encoders['categorical'].transform(X_categorical)
    
    # Extract time and target
    X_time = df['GW']
    y = df[target]
    
    # Combine processed features
    X_time_df = pd.DataFrame(X_time.values, columns=['GW'])
    X_cont_df = pd.DataFrame(X_minmax_scaled, columns=min_max_features)
    X_cat_df = pd.DataFrame(X_categorical_encoded, 
                           columns=encoders['categorical'].get_feature_names_out(categorical_features))
    X_metadata_df = pd.DataFrame(X_metadata.values, columns=metadata_features)
    X_binary_df = pd.DataFrame(X_binary.values, columns=binary_features)
    target_df = pd.DataFrame(y.values, columns=[target])
    
    X_processed = pd.concat([X_time_df, X_cont_df, X_cat_df, X_metadata_df, 
                             X_binary_df, target_df], axis=1)

    if fit:
        return X_processed, scalers, encoders
    else:
        return X_processed

def player_lag_features(gw_df, features, lags):
    """
    Create lagged features for each player in the dataframe. 
    """
    
    out_df = gw_df.copy()
    lagged_features = []
    
    # Sort the dataframe once at the beginning
    out_df = out_df.sort_values(['season_x', 'GW']).reset_index(drop=True)
    
    for feature in features:
            
        for lag in lags:
            
            lagged_feature = 'last_' + str(lag) + '_' + feature
            
            if lag == 'all':
                out_df[lagged_feature] = out_df.groupby(['season_x', 'element'])[feature]\
                    .apply(lambda x: x.cumsum() - x).reset_index(level=[0, 1], drop=True)
                
            else:
                out_df[lagged_feature] = out_df.groupby(['season_x', 'element'])[feature]\
                    .apply(lambda x: x.rolling(min_periods=1, window=lag+1).sum() - x).reset_index(level=[0, 1], drop=True)
            
            out_df[lagged_feature] = out_df[lagged_feature].round(10)
            lagged_features.append(lagged_feature)
    
    return out_df, lagged_features


def create_sequences_train(df, past_sequences=5, future_sequences=3, meta_data=[]):
    """
    Create sequences of data for each player with mapping information.
    aggressive padding strategy:
        - Creates multiple padded versions for early gameweeks
        - Adds random padding to later gameweeks to simulate missing data scenarios
        - Helps model learn to handle incomplete historical information
    
    Returns:
        X_tensor: Input sequences
        y_tensor: Target sequences  
        mapping_df: DataFrame with player/GW info for each sequence
    """
    feature_cols = sorted([col for col in df.columns if col not in meta_data])
    X_seq, y_seq = [], []
    mapping_info = []
    
    num_features = len(feature_cols)
    for player_id, stats in df.groupby(['element', 'season_x']):
        group = stats.sort_values('GW').reset_index(drop=True)
        
        # MISSING: Normal sequence creation for players with sufficient history
        for i in range(past_sequences, len(group) - future_sequences + 1):
            target_start_idx = i
            
            if target_start_idx + future_sequences - 1 >= len(group):
                continue
            
            # Take normal historical data (no padding)
            sequence_data = group.iloc[i + 1 - past_sequences:i + 1][feature_cols].values
            target = group.iloc[target_start_idx:target_start_idx + future_sequences]['total_points'].values
            
            if len(target) != future_sequences:
                continue
            
            X_seq.append(sequence_data)
            y_seq.append(target)

            prediction_gw = group.iloc[target_start_idx]['GW']
            mapping_info.append({
                    'sequence_idx': len(X_seq) - 1,
                    'element': player_id[0],
                    'season_x': player_id[1],
                    'name': group.iloc[target_start_idx]['name'],
                    'prediction_gw': prediction_gw,
                    'team_x': group.iloc[target_start_idx]['team_x'] if 'team_x' in group.columns else None,
                    'value': group.iloc[target_start_idx]['value'] if 'value' in group.columns else None,
                    'minutes': group.iloc[target_start_idx]['minutes'] if 'minutes' in group.columns else None,
                    'padding_used': 0,
                    'position_encoded': group.iloc[target_start_idx]['position_encoded'] if 'position_encoded' in group.columns else None,
                    'sequence_type': f'artificial_padding_{0}'
                })

    for player_id, stats in df.groupby(['element', 'season_x']):
        group = stats.sort_values('GW').reset_index(drop=True)
        # Add sequences with artificial padding for mid-season scenarios
        # This simulates situations where we have limited historical data due to transfers, injuries, etc.
        
        for i in range(past_sequences, min(len(group) - future_sequences + 1, 20)):  # Limit to first 20 GWs
            target_start_idx = i
            
            if target_start_idx + future_sequences - 1 >= len(group):
                continue
            
            # Create sequences with different amounts of artificial padding
            for padding_amount in [1, 2, 3]:  # Add 1, 2, or 3 steps of padding
                if i + 1 - padding_amount <= 0:
                    continue
                
                # Take less historical data and pad the beginning
                actual_history_length = past_sequences - padding_amount
                actual_data = group.iloc[i + 1 - actual_history_length:i + 1][feature_cols].values
                
                # Create padded sequence
                padded_sequence = np.zeros((past_sequences, num_features))
                padded_sequence[padding_amount:] = actual_data
                
                # Get target values
                target = group.iloc[target_start_idx:target_start_idx + future_sequences]['total_points'].values
                
                if len(target) != future_sequences:
                    continue
                
                X_seq.append(padded_sequence)
                y_seq.append(target)
                
                # Store mapping information
                prediction_gw = group.iloc[target_start_idx]['GW']
                mapping_info.append({
                    'sequence_idx': len(X_seq) - 1,
                    'element': player_id[0],
                    'season_x': player_id[1],
                    'name': group.iloc[target_start_idx]['name'],
                    'prediction_gw': prediction_gw,
                    'team_x': group.iloc[target_start_idx]['team_x'] if 'team_x' in group.columns else None,
                    'value': group.iloc[target_start_idx]['value'] if 'value' in group.columns else None,
                    'minutes': group.iloc[target_start_idx]['minutes'] if 'minutes' in group.columns else None,
                    'padding_used': padding_amount,
                    'position_encoded': group.iloc[target_start_idx]['position_encoded'] if 'position_encoded' in group.columns else None,
                    'sequence_type': f'artificial_padding_{padding_amount}'
                })
        
        # Add sequences simulating "new player" scenarios (heavy padding)
        # Take a few mid-season predictions and treat them as if the player just started
        for i in [10, 15, 20, 25]:  # Sample some mid-season gameweeks
            if i >= len(group) - future_sequences + 1:
                continue
            
            target_start_idx = i
            if target_start_idx + future_sequences - 1 >= len(group):
                continue
            
            # Create heavily padded sequences (simulating new players)
            for simulated_history in [1, 2]:  # Simulate having only 1-2 games of history
                actual_data = group.iloc[i + 1 - simulated_history:i + 1][feature_cols].values
                padding_needed = past_sequences - simulated_history
                
                padded_sequence = np.zeros((past_sequences, num_features))
                padded_sequence[padding_needed:] = actual_data
                
                target = group.iloc[target_start_idx:target_start_idx + future_sequences]['total_points'].values
                
                if len(target) != future_sequences:
                    continue
                
                X_seq.append(padded_sequence)
                y_seq.append(target)
                
                prediction_gw = group.iloc[target_start_idx]['GW']
                mapping_info.append({
                    'sequence_idx': len(X_seq) - 1,
                    'element': player_id[0],
                    'season_x': player_id[1],
                    'name': group.iloc[target_start_idx]['name'],
                    'prediction_gw': prediction_gw,
                    'team_x': group.iloc[target_start_idx]['team_x'] if 'team_x' in group.columns else None,
                    'value': group.iloc[target_start_idx]['value'] if 'value' in group.columns else None,
                    'minutes': group.iloc[target_start_idx]['minutes'] if 'minutes' in group.columns else None,
                    'padding_used': padding_needed,
                    'position_encoded': group.iloc[target_start_idx]['position_encoded'] if 'position_encoded' in group.columns else None,
                    'sequence_type': f'new_player_sim_{simulated_history}'
                })
    
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)
    mapping_df = pd.DataFrame(mapping_info)
    
    return X_tensor, y_tensor, mapping_df


#should really have train and test sequences in the same function
def create_sequences_test(df, past_sequences=5, future_sequences=3, meta_data=[], static_features=[]):
    """
    Create sequences of data for each player with mapping information.
    For early gameweeks, pad with zeros:
    - GW1 prediction: 4 zeros + GW1 data
    - GW2 prediction: 3 zeros + GW1-GW2 data  
    - GW3 prediction: 2 zeros + GW1-GW3 data
    - GW4 prediction: 1 zero + GW1-GW4 data
    - GW5+ prediction: GW(n-4) to GW(n) data (no padding)
    
    Returns:
        X_tensor: Input sequences
        y_tensor: Target sequences  
        mapping_df: DataFrame with player/GW info for each sequence
    """
    feature_cols = sorted([col for col in df.columns if col not in meta_data + static_features])
    X_seq_gw, X_seq_static, y_seq = [], [], []
    mapping_info = []
    pos_ids_seq = []
    fixdiff_ids_seq = []
    print("order of features:", feature_cols)
    print("order of static features:", static_features)

    for player_id, stats in df.groupby(['element', 'season_x']):
        group = stats.sort_values('GW').reset_index(drop=True)
        
        # Start from the first gameweek (index 0) and create sequences
        for i in range(len(group) - future_sequences + 1):
            # For predictions, we need the next gameweek(s) after position i
            target_start_idx = i
            
            # Skip if we don't have enough future data
            if target_start_idx + future_sequences - 1 >= len(group):
                continue
            
            # Determine how much historical data we have available
            available_history = i + 1

            if available_history >= past_sequences:
                # We have enough history, take the last 'past_sequences' gameweeks
                history_slice = group.iloc[i+1 - past_sequences:i+1]
                sequence_data_gw = history_slice[feature_cols].values
                static_data = history_slice.iloc[-1][static_features].values
                fixture_diff_window = history_slice['lagged_fixture_difficulty'].values
            else:
                # We need padding for early gameweeks
                actual_history = group.iloc[0:i+1]
                padding_needed = past_sequences - available_history

                sequence_data_gw = np.zeros((past_sequences, len(feature_cols)))
                sequence_data_gw[padding_needed:] = actual_history[feature_cols].values

                static_data = actual_history.iloc[-1][static_features].values 

                fixture_diff_window = np.zeros(past_sequences, dtype=int)
                fixture_diff_window[padding_needed:] = actual_history['lagged_fixture_difficulty'].astype(int).values

            # Get target values
            target = group.iloc[target_start_idx:target_start_idx + future_sequences]['total_points'].values

            if len(target) != future_sequences:
                continue

            X_seq_gw.append(sequence_data_gw)
            X_seq_static.append(static_data)
            y_seq.append(target)

            # Position id (single value) repeated across window
            pos_id = int(group.iloc[target_start_idx]['position_encoded']) if 'position_encoded' in group.columns else 0
            pos_ids_seq.append(np.full(past_sequences, pos_id, dtype=int))
            fixdiff_ids_seq.append(fixture_diff_window)
            
            # Store mapping information for the prediction gameweek
            prediction_gw = group.iloc[target_start_idx]['GW']
            mapping_info.append({
                'sequence_idx': len(X_seq_gw) - 1,
                'element': player_id[0],
                'season_x': player_id[1],
                'name': group.iloc[target_start_idx]['name'],
                'prediction_gw': prediction_gw,
                'team_x': group.iloc[target_start_idx]['team_x'] if 'team_x' in group.columns else None,
                'value': group.iloc[target_start_idx]['value'] if 'value' in group.columns else None,
                'minutes': group.iloc[target_start_idx]['minutes'] if 'minutes' in group.columns else None,
                'last_1_goals_scored': group.iloc[target_start_idx]['last_1_goals_scored'] if 'last_1_goals_scored' in group.columns else None,
                'last_1_assists': group.iloc[target_start_idx]['last_1_assists'] if 'last_1_assists' in group.columns else None,
                'padding_used': max(0, past_sequences - (i + 1)),  # Track how much padding was used
                'position_encoded': group.iloc[target_start_idx]['position_encoded'] if 'position_encoded' in group.columns else None
            })
    
    X_tensor = torch.tensor(np.array(X_seq_gw), dtype=torch.float32)
    X_static_tensor = torch.from_numpy(np.stack([np.asarray(a, dtype=np.float32) for a in X_seq_static]))
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)

    pos_ids_tensor = torch.tensor(np.array(pos_ids_seq), dtype=torch.long)          # (N, seq_len)
    fixdiff_ids_tensor = torch.tensor(np.array(fixdiff_ids_seq), dtype=torch.long)  # (N, seq_len)

    mapping_df = pd.DataFrame(mapping_info)
    return X_tensor, X_static_tensor, y_tensor, mapping_df, pos_ids_tensor, fixdiff_ids_tensor

def main():
    base_path = os.getcwd() + '/predictor/'
    file_path = os.path.join(base_path, "data/cleaned_merged_seasons.csv")
    data = pd.read_csv(file_path)

    data = data.rename(columns={
    'selected': 'chosen_by',
    })

    data["position_encoded"] = data["position"].map(position_mapping)

    data['name'] = data.name.str.replace('_\d+','')
    data['name'] = data['name'].str.replace(" ", "_").str.replace("-", "_").str.replace('_\d+','')
    data['name'] = data['name'].apply(lambda x: unidecode.unidecode(x))
    data['name'] = data['name'].str.lower()

    data = fix_gameweek_labels(data)

    data = data[data['GW'] <= 38]

    data = add_fixture_difficulty_to_dataframe(data, base_path)

    data = add_new_features(data)

    data = filter_data(data)

    data, per_GW_lagged_features = player_lag_features(data, player_features_to_lag, [1])
    data, static_features = player_lag_features(data, player_features_to_lag, ["all"])
    lagged_features = per_GW_lagged_features + static_features

    data = add_future_lagged_features(data)
    #include new lagged features
    lagged_features.extend(["lagged_was_home", "season_progress"])


    continuous_features = [col for col in lagged_features if col not in ["was_home", "total_points"]]
    meta_data = ['season_x', 'value', 'team_x', 'name', 'element', 'minutes', 'position_encoded', 'lagged_fixture_difficulty']

    seasons = np.unique(data["season_x"])
    number_of_training_seasons = len(seasons) - 2
    train_seasons = seasons[:number_of_training_seasons]
    val_seasons = seasons[number_of_training_seasons:number_of_training_seasons + 1]
    test_seasons = seasons[number_of_training_seasons + 1:]
    print("train seasons:", train_seasons)
    print("val seasons:", val_seasons)
    print("test seasons:", test_seasons)

    train = data[data["season_x"].isin(train_seasons)]
    val = data[data["season_x"].isin(val_seasons)]
    test = data[data["season_x"].isin(test_seasons)]

    train, scalers, encoders = normalize_and_encode(train, min_max_features=continuous_features, categorical_features=[], target="total_points",
                                binary_features=["was_home"], metadata_features=meta_data, fit=True)
                                #this is confusing, binary features means nothing happens and these columns are already ready
    
    val = normalize_and_encode(val, min_max_features=continuous_features, categorical_features=[], target="total_points",
                                binary_features=["was_home"], metadata_features=meta_data, fit=False,
                                scalers=scalers, encoders=encoders)
    test = normalize_and_encode(test, min_max_features=continuous_features, categorical_features=[], target="total_points",
                                binary_features=["was_home"], metadata_features=meta_data, fit=False,
                                scalers=scalers, encoders=encoders)
    
    pickle.dump(scalers, open(os.path.join(base_path, "processed_data", "scalers.pkl"), "wb"))
    pickle.dump(encoders, open(os.path.join(base_path, "processed_data", "encoders.pkl"), "wb"))


    # Update lagged_features to only include remaining features
    #remaining_lagged_features = [col for col in lagged_features if col not in removed_features['removed_columns']]
    pickle.dump(lagged_features, open(os.path.join(base_path, "processed_data", "remaining_lagged_features.pkl"), "wb"))

    extra_needed =  ['element', 'total_points', 'GW', 'season_x', 'name', 'value', 'minutes', 'position_encoded', 'lagged_fixture_difficulty']
    needed_features = extra_needed + lagged_features

    for col in [col for col in needed_features if col not in ['season_x', 'name']]:
        train[col] = train[col].astype(float)
        val[col] = val[col].astype(float)
        test[col] = test[col].astype(float)

    #also save data pre-sequences
    output_dir = os.path.join(base_path, "processed_data")
    train.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    #test for NA values
    for df in [train, val, test]:
        print(df.isna().sum())
    print(train["position_encoded"].unique())

    X_train, X_static_train, y_train, train_mapping, pos_train, fixdiff_train = create_sequences_test(
        train[needed_features], 
        past_sequences=5, 
        future_sequences=1, 
        meta_data= extra_needed,
        static_features=static_features
    )

    X_val, X_static_val, y_val, val_mapping, pos_val, fixdiff_val = create_sequences_test(
        val[needed_features], 
        past_sequences=5, 
        future_sequences=1, 
        meta_data= extra_needed,
        static_features=static_features
    )

    X_test, X_static_test, y_test, test_mapping, pos_test, fixdiff_test = create_sequences_test(
        test[needed_features], 
        past_sequences=5, 
        future_sequences=1, 
        meta_data= extra_needed,
        static_features=static_features
    )
    #check for Nan again
    for tensor in [X_train, X_static_train, y_train]:
        print(tensor.isnan().sum())

    # Save processed sequence data

    torch.save(X_train, os.path.join(output_dir, "X_train.pt"))
    torch.save(X_static_train, os.path.join(output_dir, "X_static_train.pt"))
    torch.save(y_train, os.path.join(output_dir, "y_train.pt"))
    train_mapping.to_csv(os.path.join(output_dir, "train_mapping.csv"), index=False)
    torch.save(pos_train, os.path.join(output_dir, "pos_train.pt"))
    torch.save(fixdiff_train, os.path.join(output_dir, "fixdiff_train.pt"))

    torch.save(X_val, os.path.join(output_dir, "X_val.pt"))
    torch.save(X_static_val, os.path.join(output_dir, "X_static_val.pt"))
    torch.save(y_val, os.path.join(output_dir, "y_val.pt"))
    val_mapping.to_csv(os.path.join(output_dir, "val_mapping.csv"), index=False)
    torch.save(pos_val, os.path.join(output_dir, "pos_val.pt"))
    torch.save(fixdiff_val, os.path.join(output_dir, "fixdiff_val.pt"))

    torch.save(X_test, os.path.join(output_dir, "X_test.pt"))
    torch.save(X_static_test, os.path.join(output_dir, "X_static_test.pt"))
    torch.save(y_test, os.path.join(output_dir, "y_test.pt"))
    test_mapping.to_csv(os.path.join(output_dir, "test_mapping.csv"), index=False)
    torch.save(pos_test, os.path.join(output_dir, "pos_test.pt"))
    torch.save(fixdiff_test, os.path.join(output_dir, "fixdiff_test.pt"))

    train_names = train_mapping['name'].astype(str).unique()
    name_to_idx = {n: i for i, n in enumerate(sorted(train_names))}
    unk_id = len(name_to_idx)
    name_to_idx['<unk>'] = unk_id
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    vocab_dir = os.path.join(output_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "player_name_to_idx.json"), "w") as f:
        json.dump(name_to_idx, f)
    with open(os.path.join(vocab_dir, "idx_to_player_name.json"), "w") as f:
        json.dump({str(k): v for k, v in idx_to_name.items()}, f)
    with open(os.path.join(vocab_dir, "unk_id.txt"), "w") as f:
        f.write(str(unk_id))


    def map_player_ids(mapping_df):
        mapping_df = mapping_df.sort_values('sequence_idx')
        ids = mapping_df['name'].map(lambda x: name_to_idx.get(str(x), unk_id)).to_numpy()
        assert (mapping_df['sequence_idx'].to_numpy() == np.arange(len(mapping_df))).all(), "sequence_idx not contiguous from 0"
        return torch.tensor(ids, dtype=torch.long)

    train_player_ids = map_player_ids(train_mapping)
    val_player_ids = map_player_ids(val_mapping)
    test_player_ids = map_player_ids(test_mapping)

    torch.save(train_player_ids, os.path.join(output_dir, "train_player_ids.pt"))
    torch.save(val_player_ids, os.path.join(output_dir, "val_player_ids.pt"))
    torch.save(test_player_ids, os.path.join(output_dir, "test_player_ids.pt"))
    print(f"Name vocab size (incl <unk>): {len(name_to_idx)}  unk_id={unk_id}")

if __name__ == "__main__":
    main()
    print("Preprocessing complete. Processed data saved to 'processed' directory.")