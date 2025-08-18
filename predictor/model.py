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
    def forward(self, x, mask=None):
        # x: (B,T,D)
        scores = self.proj(x).squeeze(-1)  # (B,T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B,T,1)
        return (x * w).sum(dim=1), w.squeeze(-1)

class FPLSequenceModel(nn.Module):
    def __init__(
        self,
        numeric_seq_dim,
        static_dim,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3,
        position_vocab_size=None,
        position_embed_dim=6,
        fixture_diff_vocab_size=None,
        fixture_diff_embed_dim=4,
        multitask=False
    ):
        super().__init__()
        self.use_position = position_vocab_size is not None
        self.use_fixdiff = fixture_diff_vocab_size is not None
        emb_dims = 0

        if self.use_position:
            self.position_embedding = nn.Embedding(position_vocab_size, position_embed_dim, padding_idx=0)
            emb_dims += position_embed_dim
        if self.use_fixdiff:
            self.fixdiff_embedding = nn.Embedding(fixture_diff_vocab_size, fixture_diff_embed_dim, padding_idx=0)
            emb_dims += fixture_diff_embed_dim

        self.embedding_dropout = nn.Dropout(0.1)

        lstm_input_dim = numeric_seq_dim + emb_dims

        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.locked_dropout_in = LockedDropout(dropout)
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.locked_dropout_out = LockedDropout(dropout)
        self.attn_pool = AttentionPool(hidden_dim)
        self.out_dropout = nn.Dropout(dropout * 0.5)

        fusion_dim = hidden_dim * 3 + static_dim  # attn + mean + max + static

        self.head_points = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.multitask = multitask
        if multitask:
            self.head_minutes = nn.Linear(fusion_dim, 1)
            self.head_xgi = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        seq_numeric,         # (B,T,Fn)
        static_numeric,      # (B,Fs)
        pos_ids=None,
        fixdiff_ids=None,
        mask: Optional[torch.Tensor]=None  # (B,T) boolean
    ):
        B, T, _ = seq_numeric.shape
        parts = [seq_numeric]
        if self.use_position:
            if pos_ids.dim() == 1:
                pos_ids = pos_ids.unsqueeze(1).expand(-1, T)
            parts.append(self.position_embedding(pos_ids))
        if self.use_fixdiff:
            parts.append(self.fixdiff_embedding(fixdiff_ids))
        x = torch.cat(parts, dim=-1)
        x = self.embedding_dropout(x)
        x = self.locked_dropout_in(x)
        out, _ = self.lstm(x)
        out = self.locked_dropout_out(out)

        if mask is not None:
            # ensure mask shape (B,T)
            # Replace masked positions with large negative for pooling if needed
            pass

        attn_vec, _ = self.attn_pool(out, mask)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        fused = torch.cat([attn_vec, mean_pool, max_pool, static_numeric], dim=-1)
        fused = self.out_dropout(fused)
        points = self.head_points(fused)

        if self.multitask:
            minutes = self.head_minutes(fused)
            xgi = self.head_xgi(fused)
            return points, minutes, xgi
        return points

if __name__ == "__main__":
    B = 32
    T = 5
    seq_feat_dim = 26          # per-GW numeric features
    static_feat_dim = 8        # example: season aggregates, team strength, etc.
    #SEPARATES INTO STATIC AND PER-GW FEATURES
    # Fake dataß
    seq_numeric = torch.randn(B, T, seq_feat_dim)
    static_numeric = torch.randn(B, static_feat_dim)

    # 1. Plain model (no embeddings, single-task)
    model_plain = FPLSequenceModel(
        numeric_seq_dim=seq_feat_dim,
        static_dim=static_feat_dim,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3
    )
    out_plain = model_plain(seq_numeric, static_numeric)
    print("Plain output shape:", out_plain.shape)
    print(out_plain)
    # 2. With embeddings + multitask
    position_vocab_size = 5          # e.g. 0 pad, then GK/DEF/MID/FWD
    fixture_diff_vocab_size = 8      # example buckets of fixture difficulty

    pos_ids = torch.randint(1, position_vocab_size, (B,))           # (B,) will be expanded
    fixdiff_ids = torch.randint(0, fixture_diff_vocab_size, (B, T)) # (B,T)

    model_embed = FPLSequenceModel(
        numeric_seq_dim=seq_feat_dim,
        static_dim=static_feat_dim,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3,
        position_vocab_size=position_vocab_size,
        position_embed_dim=6,
        fixture_diff_vocab_size=fixture_diff_vocab_size,
        fixture_diff_embed_dim=4,
        multitask=True
    )

    # Optional mask (all valid here)
    mask = torch.ones(B, T, dtype=torch.bool)

    points, minutes, xgi = model_embed(
        seq_numeric=seq_numeric,
        static_numeric=static_numeric,
        pos_ids=pos_ids,
        fixdiff_ids=fixdiff_ids,
        mask=mask
    )
    print("Multitask shapes:", points.shape, minutes.shape, xgi.shape)