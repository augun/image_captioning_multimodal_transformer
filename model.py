import torch
import torch.nn as nn

class TransformerCaptioningModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, patch_dim, num_heads=8, num_layers=6, max_patches=256):
        super().__init__()
        self.image_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(max_patches, embed_dim))
        self.caption_embed = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, img_patches, tgt_input):
        img_embeds = self.image_proj(img_patches)
        img_embeds += self.pos_encoder[:img_embeds.size(1)]
        img_embeds = img_embeds.permute(1, 0, 2)

        tgt_embeds = self.caption_embed(tgt_input).permute(1, 0, 2)

        memory = self.encoder(img_embeds)
        output = self.decoder(tgt_embeds, memory)
        logits = self.output_proj(output)

        return logits.permute(1, 0, 2)
