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
    

# ...existing code...
class AdvancedLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim=1,
                 num_layers=3,
                 dropout=0.3,
                 num_fc_layers=2,
                 position_vocab_size=None,
                 position_embed_dim=4,
                 player_vocab_size=None,
                 player_embed_dim=16,
                 unknown_player_index=None,
                 fixture_diff_vocab_size=None,
                 fixture_diff_embed_dim=4):
        super().__init__()
        self.use_position = position_vocab_size is not None
        self.use_player = player_vocab_size is not None
        self.use_fixdiff = fixture_diff_vocab_size is not None

        if self.use_position:
            self.position_embedding = nn.Embedding(position_vocab_size, position_embed_dim, padding_idx=0)
        else:
            position_embed_dim = 0

        if self.use_player:
            if unknown_player_index is None:
                unknown_player_index = player_vocab_size - 1
            self.unknown_player_index = unknown_player_index
            self.player_embedding = nn.Embedding(player_vocab_size, player_embed_dim)
        else:
            player_embed_dim = 0
            self.unknown_player_index = None

        if self.use_fixdiff:
            self.fixdiff_embedding = nn.Embedding(fixture_diff_vocab_size, fixture_diff_embed_dim, padding_idx=0)
        else:
            fixture_diff_embed_dim = 0

        total_input_dim = input_dim + position_embed_dim + player_embed_dim + fixture_diff_embed_dim

        self.lstm = nn.LSTM(total_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        fc_layers = []
        cur = hidden_dim
        for i in range(num_fc_layers - 1):
            nxt = max(output_dim, cur // 2)
            fc_layers += [nn.Linear(cur, nxt), nn.BatchNorm1d(nxt), nn.ReLU(), nn.Dropout(dropout)]
            cur = nxt
        fc_layers.append(nn.Linear(cur, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x_numeric, pos_ids=None, player_ids=None, fixdiff_ids=None):
        # x_numeric: (B,T,feat)
        B, T, _ = x_numeric.shape
        parts = [x_numeric]

        if self.use_position:
            if pos_ids is None:
                raise ValueError("pos_ids required")
            # pos_ids: (B,T) or (B,) -> ensure (B,T)
            if pos_ids.dim() == 1:
                pos_ids = pos_ids.unsqueeze(1).expand(-1, T)
            pos_emb = self.position_embedding(pos_ids)  # (B,T,Ep)
            parts.append(pos_emb)

        if self.use_player:
            if player_ids is None:
                raise ValueError("player_ids required")
            pid = player_ids.clone()
            pid[(pid < 0) | (pid >= self.player_embedding.num_embeddings)] = self.unknown_player_index
            pl_emb = self.player_embedding(pid).unsqueeze(1).expand(-1, T, -1)
            parts.append(pl_emb)

        if self.use_fixdiff:
            if fixdiff_ids is None:
                raise ValueError("fixdiff_ids required")
            fixdiff_emb = self.fixdiff_embedding(fixdiff_ids)  # (B,T,Ef)
            parts.append(fixdiff_emb)

        x = torch.cat(parts, dim=-1)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        return self.head(h_last)


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

    # Advanced model WITHOUT embeddings (backward compatible)
    adv_model_plain = AdvancedLSTM(input_dim=26, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.3, num_fc_layers=3)
    print("Advanced (no embeddings) output shape:", adv_model_plain(input_tensor).shape)

    # Advanced model WITH position & player embeddings
    position_vocab_size = 4          # e.g., GK, DEF, MID, FWD + variants
    player_vocab_size = 1000 + 1     # include +1 for <unk>
    unknown_idx = player_vocab_size - 1

    adv_model_embed = AdvancedLSTM(
        input_dim=26,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        dropout=0.3,
        num_fc_layers=3,
        position_vocab_size=position_vocab_size,
        position_embed_dim=6,
        player_vocab_size=player_vocab_size,
        player_embed_dim=24,
        unknown_player_index=unknown_idx
    )

    # Fake ids
    pos_ids = torch.randint(0, position_vocab_size, (32, 5))          # (batch, seq_len)
    player_ids = torch.randint(0, player_vocab_size - 1, (32,))    
    print(pos_ids.shape, player_ids.shape)   # exclude unknown for most
    player_ids[0] = 50000  # OOV example -> will be mapped to unknown

    output_with_emb = adv_model_embed(input_tensor, pos_ids=pos_ids, player_ids=player_ids)
    print("Advanced (with embeddings) output shape:", output_with_emb.shape)
    num_params_embed = sum(p.numel() for p in adv_model_embed.parameters())
    print(f"Number of parameters in advanced model (with embeddings): {num_params_embed}")