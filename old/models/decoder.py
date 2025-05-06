import torch
import torch.nn as nn

from models.encoder import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # 1. Masked self-attention (can’t peek at future tokens)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # 2. Cross-attention (decoder looks at encoder output)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # 3. Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        # Add & Norm layers after each block
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # x: [seq_len, batch_size, d_model] - decoder input (e.g. "A boy")
        # encoder_output: [src_seq_len, batch_size, d_model]

        # === 1. Masked Self-Attention (decoder to itself) ===
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # === 2. Cross-Attention (decoder to encoder) ===
        # Q = x (decoder), K = V = encoder_output
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=memory_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # === 3. Feed Forward ===
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, num_layers, max_len=5000):
        super().__init__()

        # Embedding for input tokens (captions)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Final projection to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        # tgt: [batch_size, tgt_seq_len]
        # encoder_output: [src_seq_len, batch_size, d_model]

        x = self.embedding(tgt)               # [batch_size, tgt_seq_len, d_model]
        x = self.pos_encoding(x)              # Add position info
        x = x.transpose(0, 1)                 # [tgt_seq_len, batch_size, d_model]

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)

        x = x.transpose(0, 1)                 # [batch_size, tgt_seq_len, d_model]
        logits = self.output_layer(x)         # [batch_size, tgt_seq_len, vocab_size]
        return logits
