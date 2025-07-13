import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
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
    rmse = root_mean_squared_error(y_val, predictions)
    #rmse = mse ** 0.5
    return rmse

if __name__ == "__main__":
    # Load data
    X_train, y_train = load_data("/Users/bragehs/Documents/FPL_forecast/backend/predictor/data/train_sequences.pt")
    X_val, y_val = load_data("/Users/bragehs/Documents/FPL_forecast/backend/predictor/data/val_sequences.pt")
    def reshape_data(X, y):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        # If y is 2D (samples, 1), flatten to (samples,)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.squeeze(1)
        return X, y
    X_train, y_train = reshape_data(X_train, y_train)
    X_val, y_val = reshape_data(X_val, y_val)
    print(f"Train sequences: {X_train.shape}, Targets: {y_train.shape}")

    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate model
    rmse = evaluate_model(rf_model, X_val, y_val)
    print(f"Random Forest RMSE: {rmse:.4f}")

    #2.95 RMSE