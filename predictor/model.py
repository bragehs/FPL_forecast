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
        self.input_dim = input_dim
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
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq, hidden)
        scores = self.attn(x).squeeze(-1)              # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)        # (batch, seq)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class HybridLSTMAttn(nn.Module):
    """
    LSTM -> (optional Transformer layer) -> (mean + max + attention pooled concat) -> MLP head(s).
    Also outputs predictive mean and log_var for heteroscedastic regression.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        dropout=0.3,
        use_transformer=True,
        transformer_heads=4,
        transformer_layers=1,
        emb_sizes=None,              # dict like {'element': (n_elements, 32), 'team': (n_teams, 8), 'position': (n_pos, 4)}
        seq_len=5,
        uncertainty=False
    ):
        super().__init__()
        self.uncertainty = uncertainty
        self.use_transformer = use_transformer
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.num_layers = num_layers

        self.embeddings = nn.ModuleDict()
        emb_out_dim = 0
        if emb_sizes:
            for key, (num, dim) in emb_sizes.items():
                self.embeddings[key] = nn.Embedding(num, dim)
                emb_out_dim += dim

        self.lstm_input_dim = input_dim + emb_out_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.transformer = None

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.attn_pool = AttentionPooling(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim if not uncertainty else output_dim * 2)
        )

    def forward(self, x, emb_inputs=None, mask=None):
        # x: (batch, seq, feat)
        if emb_inputs:
            emb_cat = []
            for k, tensor in emb_inputs.items():
                emb_cat.append(self.embeddings[k](tensor))  # (batch, seq, emb_dim)
            emb_cat = torch.cat(emb_cat, dim=-1)  # (batch, seq, sum_emb)
            x = torch.cat([x, emb_cat], dim=-1)
        lstm_out, _ = self.lstm(x)               # (batch, seq, hidden)

        if self.transformer:
            lstm_out = self.transformer(lstm_out)

        lstm_out = self.layer_norm(lstm_out)

        mean_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        attn_pool, _ = self.attn_pool(lstm_out, mask=mask)

        fused = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        fused = self.proj(fused)

        out = self.head(fused)
        if self.uncertainty:
            mean, log_var = out.chunk(2, dim=-1)
            return mean, log_var
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

    model = HybridLSTMAttn(
        input_dim=26,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        dropout=0.3,
        use_transformer=True,
        transformer_heads=4,
        transformer_layers=1,
        #emb_sizes={'element': (10, 32), 'team': (20, 8), 'position': (5, 4)},
        seq_len=5,
        uncertainty=False
    )

    training_output = model(input_tensor)
    print("output shape:", training_output.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")