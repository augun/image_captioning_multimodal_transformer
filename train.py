import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import load_flickr30k_captions, train_sentencepiece_tokenizer, Flickr30kDataset
from model import TransformerCaptioningModel
from inference import generate_caption
from PIL import Image

# ======= Configuration =======
BATCH_SIZE = 8
EPOCHS = 5
EMBED_DIM = 256
VOCAB_SIZE = 8000
PATCH_DIM = 3 * 16 * 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "data/flickr30k_images"
CAPTIONS_PATH = "data/flickr30k_images/results.csv"

# ======= Collate Function =======
def collate_fn(batch):
    return {
        "patches": nn.utils.rnn.pad_sequence([b["patches"] for b in batch], batch_first=True),
        "caption_input": nn.utils.rnn.pad_sequence([b["caption_input"] for b in batch], batch_first=True),
        "caption_label": nn.utils.rnn.pad_sequence([b["caption_label"] for b in batch], batch_first=True),
    }

# ======= Training Loop =======
def train(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        patches = batch["patches"].to(DEVICE)
        input_ids = batch["caption_input"].to(DEVICE)
        target_ids = batch["caption_label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(patches, input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")

# ======= Main =======
def main():
    print("🔹 Loading captions...")
    caption_dict = load_flickr30k_captions(CAPTIONS_PATH)

    print("🔹 Training SentencePiece tokenizer...")
    sp = train_sentencepiece_tokenizer(caption_dict)

    print("🔹 Preparing dataset...")
    dataset = Flickr30kDataset(IMAGE_DIR, caption_dict, sp)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("🔹 Building model...")
    model = TransformerCaptioningModel(EMBED_DIM, VOCAB_SIZE, PATCH_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")
        train(model, dataloader, optimizer, criterion)

    print("\n✅ Training complete.")

    # Inference on first image
    test_img_path = f"{IMAGE_DIR}/{dataset.filenames[0]}"
    test_img = Image.open(test_img_path).convert("RGB")
    caption = generate_caption(model, test_img, sp, DEVICE)
    print("\n🖼️ Sample Caption:")
    print(caption)

if __name__ == "__main__":
    main()
