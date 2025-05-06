import torch
import torch.nn as nn
import math

# --------------------------
# 1. Positional Encoding Layer
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix of [max_len x d_model] with sin and cos
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Different frequencies for each position
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)  # buffer = not a parameter (won’t be trained)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

# --------------------------
# 2. Encoder Layer
# --------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # ---- This is Multi-Head Attention part ----
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # ---- This is Add & Norm after attention ----
        self.norm1 = nn.LayerNorm(d_model)

        # ---- This is the Feed Forward network ----
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),  # expands the size
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)   # returns to original size
        )

        # ---- This is Add & Norm after feed forward ----
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # x shape: [seq_len, batch_size, d_model]

        # ---- Multi-head attention (self-attention) ----
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = x + self.dropout(attn_output)  # Add residual connection
        x = self.norm1(x)  # Normalize

        # ---- Feed-forward layer ----
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Add residual connection
        x = self.norm2(x)  # Normalize again

        return x

# --------------------------
# 3. Full Encoder
# --------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, ff_dim, num_layers, max_len=5000):
        super().__init__()

        # ---- Embedding layer for input tokens ----
        self.embedding = nn.Linear(input_dim, d_model)  # e.g. image patch → vector

        # ---- Positional Encoding ----
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # ---- Stack of Encoder Layers ----
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_dim)
            for _ in range(num_layers)  # N identical layers
        ])

    def forward(self, x, src_mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # add positional info

        x = x.transpose(0, 1)  # PyTorch MultiheadAttention expects [seq_len, batch_size, d_model]

        for layer in self.layers:
            x = layer(x, src_mask)

        return x  # still [seq_len, batch_size, d_model]
