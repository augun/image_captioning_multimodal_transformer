import os
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import extract_patches

VOCAB_SIZE = 8000
PATCH_SIZE = 16
MAX_LEN = 50

def load_flickr30k_captions(captions_path):
    """
    Parses the token file (pipe-delimited) and returns a dict:
    filename -> [caption1, caption2, ..., caption5]
    """
    caption_dict = {}
    with open(captions_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header if present
        for line in f:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 3:
                continue
            filename, _, caption = parts
            caption_dict.setdefault(filename, []).append(caption)
    return caption_dict

def train_sentencepiece_tokenizer(caption_dict, model_prefix="caption_tokenizer"):
    """
    Trains a SentencePiece model using all captions from the caption_dict.
    """
    all_captions = [caption for captions in caption_dict.values() for caption in captions]
    with open("captions.txt", "w") as f:
        for cap in all_captions:
            f.write(cap + "\n")
    spm.SentencePieceTrainer.train(
        input="captions.txt",
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, caption_dict, sp, patch_size=PATCH_SIZE, max_len=MAX_LEN):
        self.image_dir = image_dir
        self.sp = sp
        self.patch_size = patch_size
        self.max_len = max_len

        # Limit to a smaller subset if max_samples is set
        subset_filenames = list(caption_dict.keys())
        self.caption_dict = {k: caption_dict[k] for k in subset_filenames}
        self.filenames = subset_filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        caption = self.caption_dict[filename][0]  # Pick first of 5

        patches = extract_patches(image, self.patch_size)

        input_ids = [self.sp.bos_id()] + self.sp.encode(caption, out_type=int)
        target_ids = self.sp.encode(caption, out_type=int) + [self.sp.eos_id()]

        input_ids = input_ids[:self.max_len]
        target_ids = target_ids[:self.max_len]

        return {
            "patches": patches,
            "caption_input": torch.tensor(input_ids, dtype=torch.long),
            "caption_label": torch.tensor(target_ids, dtype=torch.long)
        }
