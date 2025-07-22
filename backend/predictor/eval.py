import torch
import torch.nn as nn
import numpy as np
from model import AdvancedLSTM, LSTMEncoderOnly
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_inference_sequence(player_data, feature_cols, max_sequences=5):
    """
    Create a single padded sequence for inference when you have variable amounts of historical data.
    
    Args:
        player_data: DataFrame with player's historical data (sorted by GW)
        feature_cols: List of feature columns to include
        max_sequences: Maximum sequence length (will pad to this length)
    
    Returns:
        padded_sequence: numpy array with shape (max_sequences, num_features)
    """
    # Get the feature data
    sequence_data = player_data[feature_cols].values
    num_features = len(feature_cols)
    
    # Create padded sequence with zeros on the left
    padded_sequence = np.zeros((max_sequences, num_features))
    
    # Determine how much padding is needed
    actual_length = min(len(sequence_data), max_sequences)
    padding_needed = max_sequences - actual_length
    
    if padding_needed > 0:
        # Left-pad with zeros
        padded_sequence[padding_needed:] = sequence_data[-actual_length:]
    else:
        # Take the last max_sequences if we have more data than needed
        padded_sequence = sequence_data[-max_sequences:]
    
    return padded_sequence

def load_model(model_path):
    """Load the trained model"""
    model_data = torch.load(model_path, map_location='cpu')
    model = AdvancedLSTM(
            input_dim=26,
            hidden_dim=model_data['hidden_dim'],
            output_dim=1,
            num_layers=model_data['num_layers'],
            num_fc_layers=model_data['num_fc_layers']
        ) 
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    return model

def analyze_padding(X_tensor):
    """
    Analyze padding in sequences and return indices for each padding level
    """
    padding_info = []
    
    for i in range(X_tensor.shape[0]):
        sequence = X_tensor[i]
        # Count rows that are all zeros (padding)
        zero_rows = (sequence == 0).all(axis=1)
        num_padding = zero_rows.sum().item()
        actual_data_rows = X_tensor.shape[1] - num_padding
        
        padding_info.append({
            'sequence_idx': i,
            'actual_data_rows': actual_data_rows,
            'padding_rows': num_padding
        })
    
    return padding_info

def get_predictions(model, X_tensor, batch_size=64):
    """Get predictions for all sequences"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            pred = model(batch)
            transformed_pred = torch.expm1(pred)
            predictions.append(pred.cpu())

    return torch.cat(predictions, dim=0)

def evaluate_by_padding_level(y_true, y_pred, padding_info):
    """Evaluate performance by padding level"""
    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    
    results = {}
    
    # Group by actual data rows
    for data_rows in range(1, 6):  # 1 to 5 data rows
        indices = [info['sequence_idx'] for info in padding_info 
                  if info['actual_data_rows'] == data_rows]
        
        if len(indices) > 0:
            y_true_subset = (y_true[indices])
            y_pred_subset = (y_pred[indices])
            
            mae = mae_fn(y_pred_subset, y_true_subset).item()
            rmse = torch.sqrt(mse_fn(y_pred_subset, y_true_subset)).item()
            
            results[data_rows] = {
                'count': len(indices),
                'mae': mae,
                'rmse': rmse,
                'percentage': (len(indices) / len(y_true)) * 100
            }
    
    return results

def evaluate_by_target_range(y_true, y_pred, ranges=[(0, 2), (2, 4), (4, 6), (6, 10), (10, float('inf'))]):
    """Evaluate performance by target value ranges"""
    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    
    results = {}
    
    for min_val, max_val in ranges:
        # Handle the case where y_true might be 2D
        if y_true.dim() > 1:
            y_true_flat = y_true.squeeze()
            y_pred_flat = y_pred.squeeze()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred
            
        if max_val == float('inf'):
            mask = y_true_flat >= min_val
            range_label = f"{min_val}+"
        else:
            mask = (y_true_flat >= min_val) & (y_true_flat < max_val)
            range_label = f"{min_val}-{max_val}"
        
        if mask.sum() > 0:
            y_true_subset = y_true_flat[mask]
            y_pred_subset = y_pred_flat[mask]
            
            mae = mae_fn(y_pred_subset, y_true_subset).item()
            rmse = torch.sqrt(mse_fn(y_pred_subset, y_true_subset)).item()
            
            results[range_label] = {
                'count': mask.sum().item(),
                'mae': mae,
                'rmse': rmse,
                'percentage': (mask.sum().item() / len(y_true_flat)) * 100,
                'mean_actual': y_true_subset.mean().item(),
                'mean_predicted': y_pred_subset.mean().item()
            }
    
    return results

def analyze_prediction_distribution(y_pred, ranges=[(0, 2), (2, 4), (4, 6), (6, 10), (10, float('inf'))]):
    """Analyze how often the model predicts values in each range"""
    if y_pred.dim() > 1:
        y_pred_flat = y_pred.squeeze()
    else:
        y_pred_flat = y_pred
    
    results = {}
    total_predictions = len(y_pred_flat)
    
    for min_val, max_val in ranges:
        if max_val == float('inf'):
            mask = y_pred_flat >= min_val
            range_label = f"{min_val}+"
        else:
            mask = (y_pred_flat >= min_val) & (y_pred_flat < max_val)
            range_label = f"{min_val}-{max_val}"
        
        count = mask.sum().item()
        percentage = (count / total_predictions) * 100
        
        results[range_label] = {
            'count': count,
            'percentage': percentage
        }
    
    return results

def main():
    # Load test data
    print("Loading test data...")
    X_test, y_test = torch.load("/Users/bragehs/Documents/FPL_forecast/backend/predictor/final_data/test_sequences.pt", weights_only=True)
    print(f"Test sequences: {X_test.shape}, Targets: {y_test.shape}")
    
    # Load best model (you may need to adjust these parameters based on your actual model)
    print("Loading best model...")
    model = load_model("/Users/bragehs/Documents/FPL_forecast/backend/predictor/best_model.pth")
    print("Model loaded successfully!")
    
    # Get predictions
    print("Generating predictions...")
    y_pred = get_predictions(model, X_test)
    
    # Overall performance
    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    
    overall_mae = mae_fn(y_pred, y_test).item()
    overall_rmse = torch.sqrt(mse_fn(y_pred, y_test)).item()
    
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    print(f"MAE:  {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"Total test samples: {len(y_test):,}")
    
    # Analyze padding
    print("\nAnalyzing padding levels...")
    padding_info = analyze_padding(X_test)
    
    # Evaluate by padding level
    print("\n" + "="*60)
    print("PERFORMANCE BY PADDING LEVEL")
    print("="*60)
    print("Data Rows | Count    | Percentage | MAE    | RMSE   | Description")
    print("-"*60)
    
    padding_results = evaluate_by_padding_level(y_test, y_pred, padding_info)
    
    for data_rows in sorted(padding_results.keys()):
        result = padding_results[data_rows]
        padding_rows = 5 - data_rows
        description = f"{padding_rows} pad rows" if padding_rows > 0 else "No padding"
        
        print(f"{data_rows:8d}  | {result['count']:7,} | {result['percentage']:8.1f}% | "
              f"{result['mae']:6.4f} | {result['rmse']:6.4f} | {description}")
    
    # Evaluate by target ranges
    print("\n" + "="*60)
    print("PERFORMANCE BY TARGET VALUE RANGE")
    print("="*60)
    print("Range  | Count    | Percentage | MAE    | RMSE   | Mean Actual | Mean Predicted")
    print("-"*75)
    
    target_results = evaluate_by_target_range(y_test, y_pred)
    
    for range_label in sorted(target_results.keys(), key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', ''))):
        result = target_results[range_label]
        print(f"{range_label:6s} | {result['count']:7,} | {result['percentage']:8.1f}% | "
              f"{result['mae']:6.4f} | {result['rmse']:6.4f} | {result['mean_actual']:9.2f} | "
              f"{result['mean_predicted']:12.2f}")
    
    # NEW: Analyze prediction distribution
    print("\n" + "="*60)
    print("MODEL PREDICTION DISTRIBUTION")
    print("="*60)
    print("How often does the model predict values in each range?")
    print("-"*60)
    
    pred_distribution = analyze_prediction_distribution(y_pred)
    
    print("Range  | Predicted Count | Predicted %")
    print("-"*40)
    
    for range_label in sorted(pred_distribution.keys(), key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', ''))):
        result = pred_distribution[range_label]
        print(f"{range_label:6s} | {result['count']:14,} | {result['percentage']:9.1f}%")
    
    # Compare actual vs predicted distribution
    print("\n" + "="*60)
    print("ACTUAL vs PREDICTED DISTRIBUTION COMPARISON")
    print("="*60)
    
    # Get actual distribution
    y_flat = y_test.squeeze() if y_test.dim() > 1 else y_test
    actual_distribution = analyze_prediction_distribution(y_flat)
    
    print("Range  | Actual Count | Actual % | Predicted Count | Predicted % | Difference")
    print("-"*75)
    
    for range_label in sorted(actual_distribution.keys(), key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', ''))):
        actual = actual_distribution[range_label]
        predicted = pred_distribution[range_label]
        diff = predicted['percentage'] - actual['percentage']
        
        print(f"{range_label:6s} | {actual['count']:11,} | {actual['percentage']:7.1f}% | "
              f"{predicted['count']:14,} | {predicted['percentage']:9.1f}% | {diff:+8.1f}%")
    
    # Additional analysis: correlation between padding and performance
    print("\n" + "="*60)
    print("PADDING ANALYSIS SUMMARY")
    print("="*60)
    
    padding_distribution = {}
    for info in padding_info:
        rows = info['actual_data_rows']
        if rows not in padding_distribution:
            padding_distribution[rows] = 0
        padding_distribution[rows] += 1
    
    total_sequences = len(padding_info)
    sequences_with_padding = sum(count for rows, count in padding_distribution.items() if rows < 5)
    
    print(f"Total sequences: {total_sequences:,}")
    print(f"Sequences with full data (5 gameweeks): {padding_distribution.get(5, 0):,}")
    print(f"Sequences with padding (1-4 gameweeks): {sequences_with_padding:,}")
    print(f"Percentage with padding: {(sequences_with_padding / total_sequences) * 100:.1f}%")
    
    # Performance difference between padded and non-padded
    if 5 in padding_results and len([r for r in padding_results.keys() if r < 5]) > 0:
        full_data_rmse = padding_results[5]['rmse']
        padded_rmse_values = [padding_results[r]['rmse'] for r in padding_results.keys() if r < 5]
        avg_padded_rmse = np.mean(padded_rmse_values)
        
        print(f"\nPerformance comparison:")
        print(f"RMSE with full data (5 gameweeks): {full_data_rmse:.4f}")
        print(f"Average RMSE with padding (1-4 gameweeks): {avg_padded_rmse:.4f}")
        print(f"Performance difference: {avg_padded_rmse - full_data_rmse:+.4f}")
    
    # Target value distribution analysis
    print("\n" + "="*60)
    print("TARGET VALUE DISTRIBUTION")
    print("="*60)
    
    y_flat = y_test.squeeze() if y_test.dim() > 1 else y_test
    print(f"Target statistics:")
    print(f"  Mean: {y_flat.mean().item():.2f}")
    print(f"  Std:  {y_flat.std().item():.2f}")
    print(f"  Min:  {y_flat.min().item():.2f}")
    print(f"  Max:  {y_flat.max().item():.2f}")
    
    # Show percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"Percentiles:")
    for p in percentiles:
        val = torch.quantile(y_flat, p/100).item()
        print(f"  {p:2d}th: {val:.2f}")
    
    # Prediction statistics
    y_pred_flat = y_pred.squeeze() if y_pred.dim() > 1 else y_pred
    print(f"\nPrediction statistics:")
    print(f"  Mean: {y_pred_flat.mean().item():.2f}")
    print(f"  Std:  {y_pred_flat.std().item():.2f}")
    print(f"  Min:  {y_pred_flat.min().item():.2f}")
    print(f"  Max:  {y_pred_flat.max().item():.2f}")

if __name__ == "__main__":
    main()