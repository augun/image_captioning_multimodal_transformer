import os
import sentencepiece as spm

DIGIT_WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

def save_digit_words_to_file(out_path="mnist_words.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        for word in DIGIT_WORDS:
            f.write(word + "\n")

def train_tokenizer(input_text="mnist_words.txt", model_prefix="files/caption_tokenizer", vocab_size=20):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)  # ✅ Ensure folder exists
    spm.SentencePieceTrainer.Train(
        input=input_text,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        pad_id=3
    )
    print("Tokenizer trained and saved as:", model_prefix + ".model")

if __name__ == "__main__":
    save_digit_words_to_file()
    train_tokenizer()
