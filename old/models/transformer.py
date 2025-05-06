import torch
import torch.nn as nn
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder

class TransformerModel(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model, num_heads, ff_dim, num_layers, max_len=5000):
        super().__init__()

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len
        )

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        encoder_output = self.encoder(src)                   # [src_seq_len, batch, d_model]
        output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)  # [batch, tgt_seq_len, vocab_size]
        return output