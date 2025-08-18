import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import FPLSequenceModel
import os
import numpy as np
import random
    

class Seq2OutputDataset(Dataset):
    def __init__(self, X_numeric, X_static, y,
                 pos_ids=None,
                 fixdiff_ids=None,
                 transform=False):
        self.X_numeric = X_numeric
        self.X_static = X_static
        self.pos_ids = pos_ids                # (N, seq_len)
        self.fixdiff_ids = fixdiff_ids        # (N, seq_len)
        if transform:
            self.y = torch.log1p(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx):
        x_numeric = self.X_numeric[idx]
        x_static = self.X_static[idx]
        y = self.y[idx]
        pos = self.pos_ids[idx] if self.pos_ids is not None else None
        fixd = self.fixdiff_ids[idx] if self.fixdiff_ids is not None else None
        # Return only what exists (keeps compatibility)
        if pos is None and fixd is None:
            return x_numeric, x_static, y
        return x_numeric, x_static, y, pos, fixd

def train_model(
        model,
        X_train_numeric,
        X_train_static,
        y_train,
        X_val_numeric,
        X_val_static,
        y_val,
        pos_ids_train=None,
        pos_ids_val=None,
        fixdiff_ids_train=None,
        fixdiff_ids_val=None,
        epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=64,
        verbose=2,
        transform=False,
        num_workers=0
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Seq2OutputDataset(
        X_train_numeric, X_train_static, y_train,
        pos_ids=pos_ids_train,
        fixdiff_ids=fixdiff_ids_train,
        transform=transform
    )
    val_dataset = Seq2OutputDataset(
        X_val_numeric, X_val_static, y_val,
        pos_ids=pos_ids_val,
        fixdiff_ids=fixdiff_ids_val,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)
    criterion = torch.nn.HuberLoss()
    mse = torch.nn.MSELoss() 
    mae = torch.nn.L1Loss()
    best_performance = float('inf')

    def forward_model(x_numeric, x_static, pos=None, fixd=None):
        kwargs = {}
        if getattr(model, 'use_position', False) and pos is not None:
            kwargs['pos_ids'] = pos
        if getattr(model, 'use_fixdiff', False) and fixd is not None:
            kwargs['fixdiff_ids'] = fixd
        try:
            return model(x_numeric, x_static, **kwargs) if kwargs else model(x_numeric, x_static)
        except TypeError:
            return model(x_numeric, x_static)

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        if verbose >= 2:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}")
        else:
            progress = train_loader
        for batch in progress:
            if len(batch) == 2:
                X_batch_numeric, X_batch_static, y_batch = batch
                pos_batch = fixd_batch = None
            else:
                X_batch_numeric, X_batch_static, y_batch, pos_batch, fixd_batch = batch
            X_batch_numeric = X_batch_numeric.to(device)
            X_batch_static = X_batch_static.to(device)
            y_batch = y_batch.to(device)
            if pos_batch is not None: pos_batch = pos_batch.to(device)
            if fixd_batch is not None: fixd_batch = fixd_batch.to(device)

            optimizer.zero_grad()
            output = forward_model(X_batch_numeric, X_batch_static, pos_batch, fixd_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch_numeric.size(0)
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
        sse = 0.0          # sum of squared errors
        mae_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    X_batch_numeric, X_batch_static, y_batch = batch
                    pos_batch = fixd_batch = None
                else:
                    X_batch_numeric, X_batch_static, y_batch, pos_batch, fixd_batch = batch
                X_batch_numeric = X_batch_numeric.to(device)
                X_batch_static = X_batch_static.to(device)
                y_batch = y_batch.to(device)
                if pos_batch is not None: pos_batch = pos_batch.to(device)
                if fixd_batch is not None: fixd_batch = fixd_batch.to(device)

                output = forward_model(X_batch_numeric, X_batch_static, pos_batch, fixd_batch)
                if transform:
                    output = torch.expm1(output)
                batch_mse = mse(output, y_batch)              # mean over batch
                batch_mae = mae(output, y_batch)

                bs = X_batch_numeric.size(0)
                sse += batch_mse.item() * bs                        # MSE * batch_size = SSE
                mae_sum += batch_mae.item() * bs
        avg_val_performance = (sse / len(val_loader.dataset)) ** 0.5
        avg_mae_loss = mae_sum / len(val_loader.dataset)
        if verbose >= 2:
            print(f"Epoch {epoch+1} validation RMSE: {avg_val_performance:.4f}")
            print(f"Epoch {epoch+1} validation MAE: {avg_mae_loss:.4f}")

        scheduler.step()
        
        if avg_val_performance < best_performance:
            best_performance = avg_val_performance
            if verbose >= 2:
                model_data = {
                    'model_state_dict': model.state_dict(),
                    'best_performance': float(best_performance),
                    'hidden_dim': int(model.hidden_dim),
                    'lstm_layers': int(model.lstm_layers),
                    'dropout': float(model.dropout)}
                torch.save(model_data, f"best_model.pth")
                print(f"Best model saved at epoch {epoch+1} with RMSE: {best_performance:.4f}")
    return best_performance

def hyperparameter_tuning(
        X_train_numeric, X_train_static, y_train,
        X_val_numeric, X_val_static, y_val,
        pos_ids_train=None, pos_ids_val=None,
        fixdiff_ids_train=None, fixdiff_ids_val=None,
        position_vocab_size=None, position_embed_dim=None,
        fixture_diff_vocab_size=None, fixture_diff_embed_dim=None,
        transform=False,
        epochs=10, n_trials=20, num_workers=0):
    """Random search for hyperparameter tuning"""
    
    # Define hyperparameter ranges
    param_ranges = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'hidden_dim': [64, 96, 128, 192, 256],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'lstm_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
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
            'lstm_layers': random.choice(param_ranges['lstm_layers']),
            'dropout': random.choice(param_ranges['dropout']),
            'batch_size': random.choice(param_ranges['batch_size']),
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Params: {params}")
        
        # Create model with sampled parameters
        model = FPLSequenceModel(
            numeric_seq_dim=X_train_numeric.shape[-1],
            static_dim=X_train_static.shape[-1],
            hidden_dim=params['hidden_dim'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout'],
            position_vocab_size=position_vocab_size,
            position_embed_dim=position_embed_dim,
            fixture_diff_vocab_size=fixture_diff_vocab_size,
            fixture_diff_embed_dim=fixture_diff_embed_dim,
        )

        val_rmse = train_model(
            model,
            X_train_numeric=X_train_numeric,
            X_train_static=X_train_static,
            y_train=y_train,
            X_val_numeric=X_val_numeric,
            X_val_static=X_val_static,
            y_val=y_val,
            pos_ids_train=pos_ids_train,
            pos_ids_val=pos_ids_val,
            fixdiff_ids_train=fixdiff_ids_train,
            fixdiff_ids_val=fixdiff_ids_val,
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