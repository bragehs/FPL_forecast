import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import Seq2SeqLSTM, LSTMEncoderOnly
import numpy as np

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
        future_fd_train, 
        X_val, 
        y_val, 
        future_fd_val,
        teacher_forcing_ratio,
        epochs=20,
        learning_rate=1e-4,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    train_dataset = Seq2OutputDataset(X_train, y_train)
    val_dataset = Seq2OutputDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = torch.nn.MSELoss() 
    performance = torch.nn.L1Loss()

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            #weights = create_loss_weights(y_batch)
            loss = criterion(output, y_batch)
            #loss = (weights * loss).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            progress.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training MSE: {avg_loss:.4f}")
        scheduler.step()
        # --- Validation loop ---
        model.eval()
        best_performance = float('inf')
        val_performance = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                #weights = create_loss_weights(y_batch)
                loss = performance(output, y_batch)
                #loss = (weights * loss).mean()
                val_performance += loss.item() * X_batch.size(0)
        avg_val_performance = val_performance / len(val_loader.dataset)
        print(f"Epoch {epoch+1} validation MAE: {avg_val_performance:.4f}")
        
        if avg_val_performance < best_performance:
            best_performance = avg_val_performance
            torch.save(model.state_dict(), f"best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with MAE: {best_performance:.4f}")
    return best_performance

def hyperparameter_tuning(X_train, y_train, future_fd_train, X_val, y_val, future_fd_val):
    learning_rates = [1e-3, 5e-3, 1e-2]
    hidden_dims = [64, 128, 256]
    best_mae = float('inf')
    best_params = None

    for lr in learning_rates:
        for hd in hidden_dims:
            print(f"\nTraining with learning_rate={lr}, hidden_dim={hd}")
            model = LSTMEncoderOnly(input_dim=26, hidden_dim=hd, output_dim=1)
            # Train for fewer epochs for tuning speed
            # Modify train_model to return best validation MAE
            val_mae = train_model(
                model,
                X_train=X_train, y_train=y_train, future_fd_train=future_fd_train,
                X_val=X_val, y_val=y_val, future_fd_val=future_fd_val,
                teacher_forcing_ratio=0.8,
                learning_rate=lr,
                epochs=10  # Fewer epochs for tuning
            )
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = {'learning_rate': lr, 'hidden_dim': hd}
    print(f"\nBest hyperparameters: {best_params}, MAE: {best_mae:.4f}")
    return best_params

if __name__ == "__main__":
    X_train, y_train, future_fd_train = torch.load("/Users/bragehs/Documents/FPL helper/Fantasy-Premier-League/data/predictor/data/train_sequences.pt", weights_only=True)
    X_val, y_val, future_fd_val = torch.load("/Users/bragehs/Documents/FPL helper/Fantasy-Premier-League/data/predictor/data/train_sequences.pt", weights_only=True) 

    print(f"Train sequences: {X_train.shape}, Targets: {y_train.shape}, Future FD: {future_fd_train.shape}")

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train, future_fd_train, X_val, y_val, future_fd_val)

    # Full training with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    model = LSTMEncoderOnly(input_dim=26, hidden_dim=best_params['hidden_dim'], output_dim=1)
    train_model(
        model,
        X_train=X_train, y_train=y_train, future_fd_train=future_fd_train,
        X_val=X_val, y_val=y_val, future_fd_val=future_fd_val,
        teacher_forcing_ratio=0.8,
        learning_rate=best_params['learning_rate'],
        epochs=100  # Full training
    )

    #Best model saved at epoch 100 with MAE: 0.9230