import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import AdvancedLSTM
import os
import numpy as np
import random
    

class Seq2OutputDataset(Dataset):
    def __init__(self, X, y, player_ids=None, transform=False):
        self.X = X
        self.player_ids = player_ids
        if transform:
            self.y = torch.log1p(y) 
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x= self.X[idx]
        y = self.y[idx]
        if self.player_ids is not None:
            player_id = self.player_ids[idx]
            return x, y, player_id
        return x, y

def train_model(
        model,
        X_train, 
        y_train, 
        X_val, 
        y_val,
        player_ids_train=None,
        player_ids_val=None,
        epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=64,
        verbose=2,
        transform=False,
        num_workers=0
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Seq2OutputDataset(X_train, y_train, player_ids=player_ids_train, transform=transform)
    val_dataset = Seq2OutputDataset(X_val, y_val, player_ids=player_ids_val, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = torch.nn.MSELoss() 
    mae = torch.nn.L1Loss()
    best_performance = float('inf')

    def forward_model(x, pid=None, pos=None):
        kwargs = {}
        if hasattr(model, 'use_player') and getattr(model, 'use_player') and pid is not None:
            kwargs['player_ids'] = pid
        try:
            return model(x, **kwargs) if kwargs else model(x)
        except TypeError:
            return model(x)
        
    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        if verbose >= 2:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}")
        else:
            progress = train_loader
        for batch in progress:
            if len(batch) > 2:
                X_batch, y_batch, pid_batch = batch
                X_batch, y_batch, pid_batch = X_batch.to(device), y_batch.to(device), pid_batch.to(device)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pid_batch = None
            optimizer.zero_grad()
            output = forward_model(X_batch, pid_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * X_batch.size(0)
            if verbose >= 2 and hasattr(progress, 'set_postfix'):
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        avg_loss = epoch_loss / len(train_loader.dataset)

        if verbose >= 2:
            if transform:
                print(f"Epoch {epoch+1} Training MSE (log): {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} Training MSE: {avg_loss:.4f}")
        # --- Validation loop ---
        model.eval()
        val_performance = 0
        mae_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) > 2:
                    X_batch, y_batch, pid_batch = batch
                    X_batch, y_batch, pid_batch = X_batch.to(device), y_batch.to(device), pid_batch.to(device)
                else:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pid_batch = None
                output = forward_model(X_batch, pid_batch)
                if transform:
                    output = torch.expm1(output)
                loss = criterion(output, y_batch)
                _mae = mae(output, y_batch)
                val_performance += np.sqrt(loss.item()) * X_batch.size(0)
                mae_loss += _mae.item() * X_batch.size(0)
        avg_val_performance = val_performance / len(val_loader.dataset)
        avg_mae_loss = mae_loss / len(val_loader.dataset)
        if verbose >= 2:
            print(f"Epoch {epoch+1} validation RMSE: {avg_val_performance:.4f}")
            print(f"Epoch {epoch+1} validation MAE: {avg_mae_loss:.4f}")


        if avg_val_performance < best_performance:
            best_performance = avg_val_performance
            if verbose >= 2:
                model_data = {
                    'model_state_dict': model.state_dict(),
                    'best_performance': float(best_performance),
                    'hidden_dim': int(model.hidden_dim),
                    'transformer_heads': int(model.transformer_heads),
                    'transformer_layers': int(model.transformer_layers)}
                torch.save(model_data, f"best_model.pth")
                print(f"Best model saved at epoch {epoch+1} with RMSE: {best_performance:.4f}")
    return best_performance

def hyperparameter_tuning(
        X_train, y_train, 
        X_val, y_val, 
        player_ids_train=None, player_ids_val=None,
        transform=False, 
        epochs=10, n_trials=20, num_workers=0):
    """Random search for hyperparameter tuning"""
    
    # Define hyperparameter ranges
    param_ranges = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'hidden_dim': [64, 96, 128, 192, 256],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'num_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
        'num_fc_layers': [1, 2, 3, 4],
        'batch_size': [32, 64, 128, 256]
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
            'num_fc_layers': random.choice(param_ranges['num_fc_layers']),
            'batch_size': random.choice(param_ranges['batch_size']),
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Params: {params}")
        
        # Create model with sampled parameters
        model = AdvancedLSTM(
            input_dim=X_train.shape[-1], 
            hidden_dim=params['hidden_dim'], 
            output_dim=1,
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_fc_layers=params['num_fc_layers']
        )     
        # Train model
        val_rmse = train_model(
            model,
            X_train=X_train, 
            y_train=y_train,
            X_val=X_val, 
            y_val=y_val, 
            player_ids_train=player_ids_train,
            player_ids_val=player_ids_val,
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            batch_size=params['batch_size'],
            epochs=epochs,
            verbose=1,
            transform=transform,
            num_workers=num_workers
        )
        
        results.append({**params, 'rmse': val_rmse})
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params
            print(f"New best RMSE: {best_rmse:.4f}")
    
    # Print top 5 results
    print(f"\nTop 5 hyperparameter combinations:")
    results.sort(key=lambda x: x['rmse'])
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. RMSE: {result['rmse']:.4f}, Params: {result}")
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    return best_params