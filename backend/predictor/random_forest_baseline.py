import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from training import create_loss_weights

def load_data(filepath):
    data = torch.load(filepath)
    X = data[0].numpy()  # Assuming the first element is features
    y = data[1].numpy()  # Assuming the second element is targets
    return X, y

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    #rmse = mse ** 0.5
    return mae

if __name__ == "__main__":
    # Load data
    X, y = load_data("/Users/bragehs/Documents/FPL helper/Fantasy-Premier-League/data/predictor/data/train_sequences.pt")
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    # If y is 2D (samples, 1), flatten to (samples,)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate model
    mae = evaluate_model(rf_model, X_val, y_val)
    print(f"Random Forest MAE: {mae:.4f}")