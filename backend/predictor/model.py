import torch
import torch.nn as nn

class LSTMEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)  # hidden: (num_layers, batch, hidden_dim)
        hidden_last = hidden[-1]       # (batch, hidden_dim)
        out = self.fc(hidden_last)     # (batch, output_dim)
        return out
    

class AdvancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=3, dropout=0.3, num_fc_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.num_fc_layers = num_fc_layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build dynamic FC layers
        self.fc_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_fc_layers):
            next_dim = current_dim // 2 if i < num_fc_layers - 1 else output_dim
            if i == num_fc_layers - 1:  # Last layer
                self.fc_layers.append(nn.Linear(current_dim, output_dim))
            else:
                self.fc_layers.append(nn.Linear(current_dim, next_dim))
                self.batch_norms.append(nn.BatchNorm1d(next_dim))
            current_dim = next_dim
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_last = hidden[-1]
        
        # Dynamic multi-layer head with batch norm
        out = hidden_last
        for i in range(self.num_fc_layers):
            out = self.fc_layers[i](out)
            if i < self.num_fc_layers - 1:  # Not the last layer
                out = self.batch_norms[i](out)
                out = self.relu(out)
                out = self.dropout(out)
        
        return out
    
class TransformerEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, nhead=4):
        super(TransformerEncoderOnly, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)                          # [B, S, H]
        x = self.transformer_encoder(x)                       # [B, S, H]
        x = x[:, -1, :]                                       # last time step
        x = self.dropout(x)
        out = self.output_layer(x)                            # [B, output_dim]
        return out

if __name__ == "__main__":
    #do some testing
    input_tensor = torch.randn(32, 5, 26)  # batch_size=32, seq_len=5, feature_dim=25
    target_tensor = torch.randn(32, 1)  # batch_size=32, output_len=3


    model = LSTMEncoderOnly(input_dim=26, hidden_dim=128, output_dim=1, num_layers=2, dropout=0.2)
    print("Model Architecture:")
    print(model.lstm.dropout)
    training_output = model(input_tensor)
    print("output shape:", training_output.shape)  

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")

    adv_model = AdvancedLSTM(input_dim=26, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.3, num_fc_layers=3)
    print("\nAdvanced Model Architecture:")
    print(adv_model.lstm.dropout)
    adv_output = adv_model(input_tensor)
    print("Advanced model output shape:", adv_output.shape)

    adv_num_params = sum(p.numel() for p in adv_model.parameters())
    print(f"Number of parameters in advanced model: {adv_num_params}")

    model = TransformerEncoderOnly(input_dim=26, hidden_dim=128, output_dim=1, num_layers=2, dropout=0.2, nhead=8)

    print("Model Architecture:")
    print(model.transformer_encoder.layers[0].dropout)

    training_output = model(input_tensor)
    print("output shape:", training_output.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")