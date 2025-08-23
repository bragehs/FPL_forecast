import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import FPLComponentModel
import os
import numpy as np
import random
from eval import expected_fpl_points

class CustomLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {"xg":1.0,"xa":1.0,"cs":1.0,"mins":1.0}

    def forward(self, preds, targets):
        # preds: dict from model
        if targets.dim() == 3 and targets.size(1) == 1:   # (B,1,5) -> (B,5)
            targets = targets.squeeze(1)
        xgTarget, xaTarget, csTarget, willPlayTarget, p60Target = targets.unbind(-1)
        # targets: dict tensors with same batch size
        loss_xg = F.mse_loss(preds["expected_goals"], xgTarget)
        loss_xa = F.mse_loss(preds["expected_assists"], xaTarget)
        loss_cs = F.binary_cross_entropy_with_logits(preds["clean_sheet_logit"], csTarget)
        # Huber for minutes
        loss_will_play = F.binary_cross_entropy_with_logits(preds["will_play"], willPlayTarget)
        loss_p_60 = F.binary_cross_entropy_with_logits(preds["p_60"], p60Target)

        total = self.weights["xg"]*loss_xg + self.weights["xa"]*loss_xa + self.weights["cs"]*loss_cs + self.weights["mins"]*loss_will_play + self.weights["mins"]*loss_p_60
        return total, {
            "loss_xg": loss_xg.detach(),
            "loss_xa": loss_xa.detach(),
            "loss_cs": loss_cs.detach(),
            "loss_will_play": loss_will_play.detach(),
            "loss_p_60": loss_p_60.detach()
        }


class Seq2OutputDataset(Dataset):
    def __init__(self, X_numeric, X_static, Y):
        self.X_numeric = X_numeric
        self.X_static = X_static
        self.y = Y

    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx):
        x_numeric = self.X_numeric[idx]
        x_static = self.X_static[idx]
        y = self.y[idx]
        return x_numeric, x_static, y

def train_model(
        model,
        X_train_numeric,
        X_train_static,
        y_train,
        X_val_numeric,
        X_val_static,
        y_val,
        epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=64,
        weights={"xg":1.0,"xa":1.0,"cs":1.0,"mins":1.0},
        verbose=2,
        num_workers=0
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Seq2OutputDataset(
        X_train_numeric, X_train_static, y_train
    )
    val_dataset = Seq2OutputDataset(
        X_val_numeric, X_val_static, y_val
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = CustomLoss(weights=weights)
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    best_performance = float('inf')

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        if verbose >= 2:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}")
        else:
            progress = train_loader
        for batch in progress:
            X_batch_numeric, X_batch_static, y_batch = batch
            X_batch_numeric, X_batch_static, y_batch = X_batch_numeric.to(device), X_batch_static.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch_numeric, X_batch_static)
            loss = criterion(output, y_batch)[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch_numeric.size(0)
            if verbose >= 2 and hasattr(progress, 'set_postfix'):
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        avg_loss = epoch_loss / len(train_loader.dataset)

        if verbose >= 2:
            print(f"\nEpoch {epoch+1} Training MSE: {avg_loss:.4f}")
        # --- Validation loop ---
        model.eval()
        sse = 0.0          # sum of squared errors
        mae_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch_numeric, X_batch_static, y_batch = batch
                X_batch_numeric, X_batch_static, y_batch = X_batch_numeric.to(device), X_batch_static.to(device), y_batch.to(device)

                output = model(X_batch_numeric, X_batch_static)
                positions = X_batch_static[:, -4:]  # last 4 static features are one-hot position
                points_pred = expected_fpl_points(output, positions)
                y_batch = y_batch.squeeze(1)
                batch_mse = mse(points_pred, y_batch)              # mean over batch
                batch_mae = mae(points_pred, y_batch)

                bs = X_batch_numeric.size(0)
                sse += batch_mse.item() * bs                        # MSE * batch_size = SSE
                mae_sum += batch_mae.item() * bs
        avg_val_performance = (sse / len(val_loader.dataset)) ** 0.5
        avg_mae_loss = mae_sum / len(val_loader.dataset)
        if verbose >= 2:
            print(f"Epoch {epoch+1} validation RMSE: {avg_val_performance:.4f}")
            print(f"Epoch {epoch+1} validation MAE: {avg_mae_loss:.4f}")

        scheduler.step(avg_mae_loss)

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
                print(f"Best model saved at epoch {epoch+1} with MAE: {best_performance:.4f}")
    return best_performance

def hyperparameter_tuning(
        X_train_numeric, X_train_static, y_train,
        X_val_numeric, X_val_static, y_val,
        epochs=10, n_trials=20, num_workers=0):
    """Random search for hyperparameter tuning"""
    
    # Define hyperparameter ranges
    param_ranges = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'hidden_dim': [64, 96, 128, 192, 256],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'lstm_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
        'weight': [{"xg":1.0,"xa":1.0,"cs":1.0,"mins":1.0},
                   {"xg":1.0,"xa":1.0,"cs":0.5,"mins":0.5},
                   {"xg":1.0,"xa":1.0,"cs":1.0,"mins":0.5},
                   {"xg":1.0,"xa":1.0,"cs":0.5,"mins":1.0}],
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
            'weight': random.choice(param_ranges['weight']),
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Params: {params}")
        
        # Create model with sampled parameters
        model = FPLComponentModel(
            numeric_seq_dim=X_train_numeric.shape[-1],
            static_dim=X_train_static.shape[-1],
            hidden_dim=params['hidden_dim'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout'],
        )

        val_rmse = train_model(
            model,
            X_train_numeric=X_train_numeric,
            X_train_static=X_train_static,
            y_train=y_train,
            X_val_numeric=X_val_numeric,
            X_val_static=X_val_static,
            y_val=y_val,
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            batch_size=params['batch_size'],
            weights=params['weight'],
            epochs=epochs,
            verbose=1,
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