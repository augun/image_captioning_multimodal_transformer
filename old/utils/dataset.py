import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import sentencepiece as spm
from datasets import load_dataset
from utils import extract_patches

VOCAB_SIZE = 8000
PATCH_SIZE = 16
MAX_LEN = 50

def load_flickr30k_dataset():
    return load_dataset("flickr30k", split='train')

def train_sentencepiece_tokenizer(dataset, model_prefix="caption_tokenizer"):
    captions = [cap['raw'] for item in dataset for cap in item['captions']]
    with open("captions.txt", "w") as f:
        for cap in captions:
            f.write(cap + "\n")
    spm.SentencePieceTrainer.train(
        input="captions.txt",
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

class Flickr30kDataset(Dataset):
    def __init__(self, dataset, sp, patch_size=PATCH_SIZE, max_len=MAX_LEN):
        self.dataset = dataset
        self.sp = sp
        self.patch_size = patch_size
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image']['path']).convert("RGB")
        caption = item['captions'][0]['raw']

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
