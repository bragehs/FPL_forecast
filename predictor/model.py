import torch
import torch.nn as nn
import math
from typing import Optional

class LockedDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: (B,T,F)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p).div_(1 - self.p)
        return x * mask

class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        # x: (B,T,D)
        scores = self.proj(x).squeeze(-1)  # (B,T)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B,T,1)
        return (x * w).sum(dim=1), w.squeeze(-1)

class FPLComponentModel(nn.Module):
    """
    Outputs:
      expected_goals (>=0)
      expected_assists (>=0)
      clean_sheet_logit (raw logit; apply sigmoid)
      minutes (>=0)
    """
    def __init__(
        self,
        numeric_seq_dim,
        static_dim,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.locked_dropout_in = LockedDropout(dropout)
        self.lstm = nn.LSTM(
            numeric_seq_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.locked_dropout_out = LockedDropout(dropout)
        self.attn_pool = AttentionPool(hidden_dim)
        self.out_dropout = nn.Dropout(dropout * 0.5)
        fusion_dim = hidden_dim * 3 + static_dim  # attn + mean + max + static
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(hidden_dim, 5)

    def forward(self, seq_numeric, static_numeric):
        out, _ = self.lstm(self.locked_dropout_in(seq_numeric))
        out = self.locked_dropout_out(out)
        attn_vec, _ = self.attn_pool(out)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        fused = torch.cat([attn_vec, mean_pool, max_pool, static_numeric], dim=-1)
        fused = self.out_dropout(fused)
        z = self.backbone(fused)
        raw = self.head(z)  # (B,5)
        # Split & apply activations
        raw_xg, raw_xa, cs_logit, p_play, p_60 = torch.unbind(raw, dim=-1)
        pred_xg = torch.nn.functional.softplus(raw_xg)
        pred_xa = torch.nn.functional.softplus(raw_xa)
        return {
            "expected_goals": pred_xg,
            "expected_assists": pred_xa,
            "clean_sheet_logit": cs_logit,   # apply sigmoid outside if needed
            "will_play": p_play,
            "p_60": p_60,
        }

if __name__ == "__main__":
    B,T = 8,5
    seq_feat_dim=26
    static_feat_dim=10
    model = FPLComponentModel(seq_feat_dim, static_feat_dim)
    dummy_seq = torch.randn(B,T,seq_feat_dim)
    dummy_static = torch.randn(B,static_feat_dim)
    out = model(dummy_seq, dummy_static)
    for k,v in out.items():
        print(k, v.shape)