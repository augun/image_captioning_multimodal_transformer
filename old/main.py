
import torch
from models.transformer import TransformerModel
from utils.dataset import get_dataloader
from utils.masking import generate_square_subsequent_mask

# Dummy configs
batch_size = 2
src_seq_len = 16     # Image patches
input_dim = 256      # Patch embedding size
tgt_seq_len = 10     # Caption length
vocab_size = 1000

d_model = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
max_len = 100

# Create dummy model
model = TransformerModel(
    input_dim=input_dim,
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers,
    max_len=max_len
)

# Load dataloader
train_loader = get_dataloader(batch_size=batch_size)

# Loop through dataset (image, caption_input, caption_label)
for image_patches, caption_input, caption_label in train_loader:
    tgt_mask = generate_square_subsequent_mask(caption_input.size(1)).to(caption_input.device)

    # Forward pass
    logits = model(image_patches, caption_input, tgt_mask=tgt_mask)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, vocab_size)
    caption_label = caption_label.view(-1)
    loss = loss_fn(logits, caption_label)

    print("Loss:", loss.item())
    break


