# 🖼️ Image Captioning with Multimodal Transformer (Flickr30k)

This project implements a multimodal image captioning system using a full encoder-decoder Transformer architecture, inspired by the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It generates natural language descriptions of images using the **Flickr30k** dataset.

## 📦 Dataset

This project uses the **Flickr30k** dataset, which includes ~31,000 images with five captions per image.

### 📥 Download Instructions

1. Go to: [https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?resource=download](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?resource=download)
2. Download and extract the ZIP.
3. Place the contents into this structure:

image_captioning_project/ 
└── data/ 
├── flickr30k_images/ # ~31K JPG images 
└── results.csv # Captions file (renamed from results_20130124.token)


> Note: `results.csv` must be in pipe-delimited format:  
> `image_name| caption_number| caption_text`

## 🧱 Project Structure

image_captioning_project/ 
├── dataset.py # Loads images + captions and prepares data 
├── model.py # Transformer encoder-decoder architecture 
├── inference.py # Caption generation function 
├── train.py # Training script (entry point) 
├── utils.py # Image patch extraction 
├── requirements.txt # Dependencies 
├── README.md # This file 
└── data/ # (Ignored by Git) Contains local images and captions


## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

python train.py

This will:

Load local images + captions

Train a SentencePiece tokenizer

Train a full Transformer captioning model

Generate a caption for one sample image

🖼️ Sample Caption:
Two young guys with shaggy hair look at their hands while hanging out in the yard.

🛠️ Model Details
Encoder Input: Image converted to 16x16 patches and projected into embedding space

Decoder Input: Captions tokenized with SentencePiece, with BOS/EOS markers

Architecture: Full encoder-decoder Transformer (PyTorch nn.Transformer)

Loss Function: CrossEntropyLoss with ignore_index=0

Tokenizer: SentencePiece with 8000 vocabulary size

📌 Notes
The model is trained only on the first caption for each image, but you can extend it to use all 5.

.gitignore should exclude the data/ folder to avoid uploading large files.

