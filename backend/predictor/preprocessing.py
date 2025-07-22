import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import smogn
from sklearn.metrics.pairwise import cosine_similarity

categorical_features = [
    'position',           # Player position (GK, DEF, MID, FWD)
]

binary_features = [
    'was_home',          # Home/Away (boolean)
]

min_max_features = [
    'bonus', 'minutes','fixture_difficulty', 'yellow_cards', 'red_cards', 
    'own_goals', 'saves', 'goals_conceded','clean_sheets', 'penalties_missed',
    'penalties_saved', 'assists', 'goals_scored', 'season_progress', 'expected_goals',
    'expected_assists', 'expected_goals_conceded','creativity', 'influence',
     'threat', 'ict_index', 'bps',
    # Rolling features
   # 'total_points_rolling_3', 'total_points_rolling_5',
    #'goals_rolling_3', 'goals_rolling_5',
   # 'assists_rolling_3', 'assists_rolling_5',
   # 'minutes_rolling_3', 'minutes_rolling_5',
   # 'points_std_3', 'points_std_5',
   # 'clean_sheets_rolling_3', 'clean_sheets_rolling_5',

    # Streak features
   # 'points_streak', 'games_without_return', 'starts_streak',
    

]


metadata_features = [
    'name',                    # Player name (use for grouping/analysis)
    'element',                 # Player ID (use for grouping sequences)
    'team_x',                 
    'season_x', 
]

time_feature = 'round'
target = 'total_points'  # Target variable for prediction


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


def filter_data(df): 
    df.loc[df['position'].isin(['GK', 'GKP']), 'position'] = 'GK'
    df.loc[df['total_points'] < 0, 'total_points'] = 0
    df.loc[df['fixture_difficulty'] == 1, 'fixture_difficulty'] = 2
    df = df[df['position'].isin(['GK', 'MID', 'DEF', 'FWD'])]
    df = df[df['minutes'] > 0].copy()
    df['season_progress'] = df['GW'] / df['GW'].max()
    df = df.sort_values(['season_x', 'GW']).reset_index(drop=True)
    

    return df

def oversample_sequences(X_tensor, y_tensor, target_percentile=0.8, n_synthetic=1000):
    """
    Oversample sequences directly after they've been created
    Focus on high-performing sequences (rare cases)
    """
    # Convert tensors to numpy for easier manipulation
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()
    
    # Find high-performing sequences to oversample
    target_threshold = np.percentile(y_np, target_percentile * 100)
    high_performing_mask = y_np.squeeze() >= target_threshold
    
    high_X = X_np[high_performing_mask]
    high_y = y_np[high_performing_mask]
    
    print(f"Found {len(high_X)} high-performing sequences (target >= {target_threshold:.1f})")
    
    if len(high_X) == 0:
        return X_tensor, y_tensor
    
    # Generate synthetic sequences
    synthetic_X = []
    synthetic_y = []
    
    for i in range(n_synthetic):
        # Randomly select a high-performing sequence as base
        idx = np.random.randint(0, len(high_X))
        base_X = high_X[idx].copy()
        base_y = high_y[idx].copy()
        
        # Add controlled noise to features
        # Noise level: 2-5% of feature standard deviation
        for seq_idx in range(base_X.shape[0]):  # For each timeframe in sequence
            for feat_idx in range(base_X.shape[1]):  # For each feature
                feature_values = X_np[:, seq_idx, feat_idx]  # All values for this feature
                noise_std = np.std(feature_values) * np.random.uniform(0.02, 0.05)
                noise = np.random.normal(0, noise_std)
                base_X[seq_idx, feat_idx] += noise
        
        # Add small noise to target
        target_noise = np.random.normal(0, np.std(y_np) * 0.03)
        base_y += target_noise
        base_y = np.clip(base_y, 0, np.max(y_np))  # Keep in valid range
        
        synthetic_X.append(base_X)
        synthetic_y.append(base_y)
    
    # Convert back to tensors and combine
    synthetic_X_tensor = torch.tensor(np.array(synthetic_X), dtype=torch.float32)
    synthetic_y_tensor = torch.tensor(np.array(synthetic_y), dtype=torch.float32)
    
    # Combine original and synthetic
    X_combined = torch.cat([X_tensor, synthetic_X_tensor], dim=0)
    y_combined = torch.cat([y_tensor, synthetic_y_tensor], dim=0)
    
    print(f"Original sequences: {len(X_tensor)}")
    print(f"Synthetic sequences: {len(synthetic_X_tensor)}")
    print(f"Total sequences: {len(X_combined)}")
    
    return X_combined, y_combined


def split_data(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df



def preprocess_data(df, scalers=None, encoders=None, fit=False):
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
    target_df = pd.DataFrame(y.values, columns=[target])
    
    X_processed = pd.concat([X_time_df, X_cont_df, X_cat_df, X_metadata_df,
                             target_df], axis=1)
    
    if fit:
        return X_processed, scalers, encoders
    else:
        return X_processed
    

def create_sequences(df, past_sequences=5, future_sequences=3, min_sequences=1):
    """
    Create sequences of data for each player based on the Gameweek (GW).
    Supports variable-length sequences with left-padding for flexibility.
    
    Args:
        df: DataFrame with player data
        past_sequences: Maximum number of past gameweeks to use (default: 5)
        future_sequences: Number of future gameweeks to predict (default: 3)
        min_sequences: Minimum number of past gameweeks required (default: 1)
    
    Returns:
        X_tensor: Input sequences with left-padding for shorter sequences
        y_tensor: Target sequences
    """
    # Sort to ensure features are always in same order
    feature_cols = sorted([col for col in df.columns if col not in ['total_points', 'GW', 'element', 'name', 'season_x', 'team_x']])
    X_seq, y_seq = [], []
    
    num_features = len(feature_cols)

    for player_id, stats in df.groupby(['element', 'season_x']):
        group = stats.sort_values('GW').reset_index(drop=True)
        
        # Iterate through the group to create sequences
        for i in range(min_sequences - 1, len(group) - future_sequences):
            # Determine the actual number of past gameweeks available for this sequence
            actual_past_available = min(i + 1, past_sequences)
            
            # Start index for actual sequence data
            start_idx_actual = i - actual_past_available + 1
            
            sequence_data = group.iloc[start_idx_actual : i + 1][feature_cols].values
            
            # Create the final sequence
            if len(sequence_data) < past_sequences:
                # Need padding - create padded sequence with zeros on the left
                padded_sequence = np.zeros((past_sequences, num_features))
                padding_needed = past_sequences - len(sequence_data)
                padded_sequence[padding_needed:] = sequence_data
            else:
                # No padding needed - use the last 'past_sequences' rows
                padded_sequence = sequence_data[-past_sequences:]
            
            # Get target and future fixture difficulty
            target_start_idx = i + 1
            target = group.iloc[target_start_idx : target_start_idx + future_sequences]['total_points'].values
            
            # Ensure target and future_fd have the correct length
            if len(target) != future_sequences:
                continue
                
            X_seq.append(padded_sequence)
            y_seq.append(target)
    
    # Convert to numpy arrays
    X_np = np.array(X_seq)
    y_np = np.array(y_seq)

    # Convert to tensors
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    return X_tensor, y_tensor


def main():
    base_path = os.getcwd() + "/backend/predictor/"
    #backend_root = os.path.dirname(base_path)

    file_path = os.path.join(base_path, "data/cleaned_merged_seasons.csv")
    df = pd.read_csv(file_path)
    df = df[df['season_x'].isin(['2024-25', '2023-24', '2022-23'])].copy() #only seasons with xG

    df_enhanced = add_fixture_difficulty_to_dataframe(df, base_path)

    df_filtered = filter_data(df_enhanced)

    train_df, val_df, test_df = split_data(df_filtered)

    # Preprocess data
    processed_train_df, scalers, encoders = preprocess_data(train_df, fit=True)
    processed_val_df = preprocess_data(val_df, scalers=scalers, encoders=encoders, fit=False)
    processed_test_df = preprocess_data(test_df, scalers=scalers, encoders=encoders, fit=False)

    # Oversample training data
    #processed_train_df = oversample_data(processed_train_df[:100], target_col='total_points')

    X_train, y_train = create_sequences(
    processed_train_df, 
    past_sequences=5, 
    future_sequences=1, 
    min_sequences=1 
)
    X_val, y_val = create_sequences(
    processed_val_df, 
    past_sequences=5, 
    future_sequences=1, 
    min_sequences=1
)
    X_test, y_test = create_sequences(
    processed_test_df, 
    past_sequences=5, 
    future_sequences=1, 
    min_sequences=1
)
    # Oversample high-performing sequences
    X_train, y_train = oversample_sequences(X_train, y_train, 
                                            target_percentile=0.85,
                                            n_synthetic=10000)

    print(f"Train sequences: {X_train.shape}, Targets: {y_train.shape}")
    print(f"Val sequences: {X_val.shape}, Targets: {y_val.shape}")
    print(f"Test sequences: {X_test.shape}, Targets: {y_test.shape}")

    # Save processed data
    torch.save((X_train, y_train), base_path + 'final_data/train_sequences.pt')
    torch.save((X_val, y_val), base_path + 'final_data/val_sequences.pt')
    torch.save((X_test, y_test), base_path + 'final_data/test_sequences.pt')

    print("Data preparation complete. Sequences saved to disk.")


if __name__ == "__main__":
    main()



