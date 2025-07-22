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

def oversample_sequences_balanced(X_tensor, y_tensor, n_synthetic_per_percentile=200, n_percentiles=10):
    """
    Oversample sequences to create a balanced distribution across percentiles
    
    Args:
        X_tensor: Input sequences
        y_tensor: Target values
        n_synthetic_per_percentile: Number of synthetic samples to generate per percentile
        n_percentiles: Number of percentile bins to create (default: 10 for deciles)
    
    Returns:
        X_combined: Original + synthetic sequences
        y_combined: Original + synthetic targets
    """
    # Convert tensors to numpy for easier manipulation
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy().squeeze()
    
    # Calculate percentile boundaries
    percentiles = np.linspace(0, 100, n_percentiles + 1)
    percentile_bounds = np.percentile(y_np, percentiles)
    
    print(f"Creating balanced distribution across {n_percentiles} percentiles:")
    for i in range(len(percentile_bounds) - 1):
        print(f"  P{i*10}-{(i+1)*10}: {percentile_bounds[i]:.1f} - {percentile_bounds[i+1]:.1f}")
    
    synthetic_X = []
    synthetic_y = []
    
    # Generate synthetic samples for each percentile bin
    for i in range(len(percentile_bounds) - 1):
        lower_bound = percentile_bounds[i]
        upper_bound = percentile_bounds[i + 1]
        
        # Find samples in this percentile range
        if i == len(percentile_bounds) - 2:  # Last bin, include upper bound
            mask = (y_np >= lower_bound) & (y_np <= upper_bound)
        else:
            mask = (y_np >= lower_bound) & (y_np < upper_bound)
        
        percentile_X = X_np[mask]
        percentile_y = y_np[mask]
        
        original_count = len(percentile_X)
        print(f"  Percentile {i*10}-{(i+1)*10}: {original_count} original samples")
        
        if original_count == 0:
            continue
        
        # Generate synthetic samples for this percentile
        for j in range(n_synthetic_per_percentile):
            # Randomly select a sample from this percentile as base
            idx = np.random.randint(0, len(percentile_X))
            base_X = percentile_X[idx].copy()
            base_y = percentile_y[idx].copy()
            
            # Add controlled noise to features
            for seq_idx in range(base_X.shape[0]):  # For each timeframe in sequence
                for feat_idx in range(base_X.shape[1]):  # For each feature
                    # Use feature values from the same percentile for noise calculation
                    feature_values = percentile_X[:, seq_idx, feat_idx]
                    if len(feature_values) > 1:
                        noise_std = np.std(feature_values) * np.random.uniform(0.02, 0.08)
                    else:
                        # Fallback to global feature values if percentile has only 1 sample
                        feature_values = X_np[:, seq_idx, feat_idx]
                        noise_std = np.std(feature_values) * np.random.uniform(0.02, 0.08)
                    
                    noise = np.random.normal(0, noise_std)
                    base_X[seq_idx, feat_idx] += noise
            
            # Add small noise to target while keeping it within percentile bounds
            target_noise_std = (upper_bound - lower_bound) * 0.1  # 10% of percentile range
            target_noise = np.random.normal(0, target_noise_std)
            base_y += target_noise
            
            # Clip to stay within percentile bounds (with small tolerance)
            tolerance = (upper_bound - lower_bound) * 0.05
            base_y = np.clip(base_y, 
                           max(0, lower_bound - tolerance), 
                           min(np.max(y_np), upper_bound + tolerance))
            
            synthetic_X.append(base_X)
            synthetic_y.append(base_y)
    
    if not synthetic_X:
        print("No synthetic samples generated!")
        return X_tensor, y_tensor
    
    # Convert back to tensors and combine
    synthetic_X_tensor = torch.tensor(np.array(synthetic_X), dtype=torch.float32)
    synthetic_y_tensor = torch.tensor(np.array(synthetic_y), dtype=torch.float32).unsqueeze(-1)
    
    # Combine original and synthetic
    X_combined = torch.cat([X_tensor, synthetic_X_tensor], dim=0)
    y_combined = torch.cat([y_tensor, synthetic_y_tensor], dim=0)
    
    print(f"\nBalanced oversampling results:")
    print(f"Original sequences: {len(X_tensor)}")
    print(f"Synthetic sequences: {len(synthetic_X_tensor)}")
    print(f"Total sequences: {len(X_combined)}")
    
    # Analyze final distribution
    analyze_distribution(y_combined.squeeze().numpy(), "Final Distribution")
    
    return X_combined, y_combined

def analyze_distribution(y_values, title="Distribution"):
    """Analyze and print distribution statistics"""
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_values = np.percentile(y_values, percentiles)
    
    print(f"\n{title}:")
    for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
        count = len(y_values[y_values <= val])
        if i == 0:
            prev_count = 0
        else:
            prev_count = len(y_values[y_values <= percentile_values[i-1]])
        
        bin_count = count - prev_count
        print(f"  P{p:2d}: {val:5.1f} ({bin_count:4d} samples)")

def oversample_sequences_adaptive(X_tensor, y_tensor, target_samples_per_bin=500, n_bins=10, min_original_samples=10):
    """
    Adaptive oversampling that brings all bins to the same sample count
    
    Args:
        X_tensor: Input sequences
        y_tensor: Target values
        target_samples_per_bin: Target number of samples per bin
        n_bins: Number of bins to create
        min_original_samples: Minimum original samples required in bin to generate synthetic ones
    """
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy().squeeze()
    
    # Create equal-width bins
    y_min, y_max = y_np.min(), y_np.max()
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    
    print(f"Adaptive oversampling with {n_bins} bins, target: {target_samples_per_bin} samples per bin")
    
    synthetic_X = []
    synthetic_y = []
    
    for i in range(len(bin_edges) - 1):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        
        # Find samples in this bin
        if i == len(bin_edges) - 2:  # Last bin
            mask = (y_np >= lower_bound) & (y_np <= upper_bound)
        else:
            mask = (y_np >= lower_bound) & (y_np < upper_bound)
        
        bin_X = X_np[mask]
        bin_y = y_np[mask]
        original_count = len(bin_X)
        
        # Calculate how many synthetic samples needed
        samples_needed = max(0, target_samples_per_bin - original_count)
        
        print(f"  Bin {i+1} [{lower_bound:.1f}-{upper_bound:.1f}]: {original_count} original, generating {samples_needed}")
        
        if samples_needed == 0 or original_count == 0:
            continue
        
        # Skip bins with too few original samples
        if original_count < min_original_samples:
            print(f"    Skipping bin {i+1}: only {original_count} original samples (< {min_original_samples} minimum)")
            continue
        
        # Generate synthetic samples for this bin
        for j in range(samples_needed):
            # Randomly select a sample from this bin as base
            idx = np.random.randint(0, len(bin_X))
            base_X = bin_X[idx].copy()
            base_y = bin_y[idx].copy()
            
            # Add noise (similar to previous function)
            for seq_idx in range(base_X.shape[0]):
                for feat_idx in range(base_X.shape[1]):
                    if len(bin_X) > 1:
                        feature_values = bin_X[:, seq_idx, feat_idx]
                        noise_std = np.std(feature_values) * np.random.uniform(0.03, 0.1)
                    else:
                        feature_values = X_np[:, seq_idx, feat_idx]
                        noise_std = np.std(feature_values) * np.random.uniform(0.03, 0.1)
                    
                    noise = np.random.normal(0, noise_std)
                    base_X[seq_idx, feat_idx] += noise
            
            # Add noise to target within bin bounds
            bin_range = upper_bound - lower_bound
            target_noise = np.random.normal(0, bin_range * 0.1)
            base_y += target_noise
            base_y = np.clip(base_y, lower_bound, upper_bound)
            
            synthetic_X.append(base_X)
            synthetic_y.append(base_y)
    
    if not synthetic_X:
        return X_tensor, y_tensor
    
    # Combine results
    synthetic_X_tensor = torch.tensor(np.array(synthetic_X), dtype=torch.float32)
    synthetic_y_tensor = torch.tensor(np.array(synthetic_y), dtype=torch.float32).unsqueeze(-1)
    
    X_combined = torch.cat([X_tensor, synthetic_X_tensor], dim=0)
    y_combined = torch.cat([y_tensor, synthetic_y_tensor], dim=0)
    
    print(f"\nAdaptive oversampling results:")
    print(f"Original: {len(X_tensor)}, Synthetic: {len(synthetic_X_tensor)}, Total: {len(X_combined)}")
    
    analyze_distribution(y_combined.squeeze().numpy(), "Final Balanced Distribution")
    
    return X_combined, y_combined

def undersample_sequences_percentile(X_tensor, y_tensor, target_percentile=20, target_samples=1000):
    """
    Undersample sequences in a specific percentile range
    
    Args:
        X_tensor: Input sequences
        y_tensor: Target values
        target_percentile: Percentile threshold below which to undersample (default: 20 = bottom 20%)
        target_samples: Target number of samples to keep in the undersampled percentile
    
    Returns:
        X_combined: Undersampled sequences
        y_combined: Undersampled targets
    """
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy().squeeze()
    
    # Calculate percentile threshold
    percentile_threshold = np.percentile(y_np, target_percentile)
    
    # Split data into percentile groups
    low_percentile_mask = y_np <= percentile_threshold
    high_percentile_mask = y_np > percentile_threshold
    
    low_X = X_np[low_percentile_mask]
    low_y = y_np[low_percentile_mask]
    high_X = X_np[high_percentile_mask]
    high_y = y_np[high_percentile_mask]
    
    original_low_count = len(low_X)
    original_high_count = len(high_X)
    
    print(f"Undersampling bottom {target_percentile}% (scores ≤ {percentile_threshold:.1f}):")
    print(f"  Original low percentile samples: {original_low_count}")
    print(f"  Original high percentile samples: {original_high_count}")
    
    # Undersample the low percentile if it has more samples than target
    if original_low_count > target_samples:
        # Randomly select target_samples from low percentile
        indices = np.random.choice(original_low_count, target_samples, replace=False)
        undersampled_low_X = low_X[indices]
        undersampled_low_y = low_y[indices]
        print(f"  Undersampled to: {len(undersampled_low_X)} samples")
    else:
        undersampled_low_X = low_X
        undersampled_low_y = low_y
        print(f"  No undersampling needed (already ≤ {target_samples} samples)")
    
    # Combine undersampled low percentile with all high percentile samples
    X_combined_np = np.concatenate([undersampled_low_X, high_X], axis=0)
    y_combined_np = np.concatenate([undersampled_low_y, high_y], axis=0)
    
    # Convert back to tensors
    X_combined = torch.tensor(X_combined_np, dtype=torch.float32)
    y_combined = torch.tensor(y_combined_np, dtype=torch.float32).unsqueeze(-1)
    
    print(f"\nUndersampling results:")
    print(f"Original total: {len(X_tensor)}")
    print(f"Final total: {len(X_combined)}")
    print(f"Removed: {len(X_tensor) - len(X_combined)} samples")
    
    analyze_distribution(y_combined.squeeze().numpy(), "Distribution After Undersampling")
    
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
    
    X_train, y_train = undersample_sequences_percentile(
        X_train, y_train, 
        target_percentile=60, 
        target_samples=4000
    )

    # Oversample high-performing sequences
    X_train, y_train = oversample_sequences_adaptive(
        X_train, y_train, 
        target_samples_per_bin=3000,  
        n_bins=10,
        min_original_samples=50  # Only create synthetic data if bin has at least 50 original samples
    )

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



