import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import LSTMEncoderOnly, AdvancedLSTM, TransformerEncoderOnly
import os
import numpy as np
import random

def create_loss_weights(data, alpha=4.0, percentile=0.95):
    weight_cap = torch.quantile(data, percentile)
    mean = data.mean()
    dev = (data - mean).abs()
    weights = 1 + alpha * dev / dev.max()
    weights = torch.clamp(weights, max=weight_cap)
    return weights

class Seq2SeqDataset(Dataset):
    def __init__(self, X, y, future_fd):
        self.X = X
        self.y = y
        self.future_fd = future_fd

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.future_fd[idx]
    

class Seq2OutputDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(
        model,
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    train_dataset = Seq2OutputDataset(X_train, y_train)
    val_dataset = Seq2OutputDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = torch.nn.MSELoss() 
    mae = torch.nn.L1Loss()
    best_performance = float('inf')
    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}")
        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            #weights = create_loss_weights(y_batch)
            loss = criterion(output, y_batch)
            #loss = (weights * loss).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * X_batch.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training MSE: {avg_loss:.4f}")
        # --- Validation loop ---
        model.eval()
        val_performance = 0
        mae_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                #weights = create_loss_weights(y_batch)
                loss = criterion(output, y_batch)
                _mae = mae(output, y_batch)
                #loss = (weights * loss).mean()
                val_performance += np.sqrt(loss.item()) * X_batch.size(0)
                mae_loss += _mae.item() * X_batch.size(0)
        avg_val_performance = val_performance / len(val_loader.dataset)
        avg_mae_loss = mae_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1} validation RMSE: {avg_val_performance:.4f}")
        print(f"Epoch {epoch+1} validation MAE: {avg_mae_loss:.4f}")


        if avg_val_performance < best_performance:
            best_performance = avg_val_performance
            torch.save(model.state_dict(), f"best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with RMSE: {best_performance:.4f}")
    return best_performance

def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=20):
    """Random search for hyperparameter tuning"""
    
    # Define hyperparameter ranges
    param_ranges = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'hidden_dim': [64, 96, 128, 192, 256],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'num_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
        'num_fc_layers': [1, 2, 3, 4]
    }
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    print(f"Running random search with {n_trials} trials...")
    
    for trial in range(n_trials):
        # Random sample from each hyperparameter range
        params = {
            'learning_rate': random.choice(param_ranges['learning_rate']),
            'hidden_dim': random.choice(param_ranges['hidden_dim']),
            'weight_decay': random.choice(param_ranges['weight_decay']),
            'num_layers': random.choice(param_ranges['num_layers']),
            'dropout': random.choice(param_ranges['dropout']),
            'num_fc_layers': random.choice(param_ranges['num_fc_layers'])
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Params: {params}")
        
        # Create model with sampled parameters
        model = AdvancedLSTM(
            input_dim=26, 
            hidden_dim=params['hidden_dim'], 
            output_dim=1,
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_fc_layers=params['num_fc_layers']
        )
        
        # Train model1'
        try:
            val_rmse = train_model(
                model,
                X_train=X_train, 
                y_train=y_train,
                X_val=X_val, 
                y_val=y_val, 
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                epochs=10  # Fewer epochs for tuning speed
            )
            
            results.append({**params, 'rmse': val_rmse})
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = params
                print(f"New best RMSE: {best_rmse:.4f}")
                
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    
    # Print top 5 results
    print(f"\nTop 5 hyperparameter combinations:")
    results.sort(key=lambda x: x['rmse'])
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. RMSE: {result['rmse']:.4f}, Params: {result}")
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    return best_params