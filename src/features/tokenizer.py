# src/features/tokenizer.py

import pandas as pd
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_processed_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found at {path}")

    df = pd.read_csv(path)

    # 🔥 CRITICAL FIX
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    return df


def create_tokenizer(texts, vocab_size: int):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer, texts, max_len: int):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )
    return padded


def save_tokenizer(tokenizer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✅ Tokenizer saved to: {path}")


def prepare_features(data_path: str, vocab_size: int, max_len: int, tokenizer_path: str):
    df = load_processed_data(data_path)

    texts = df["text"].values
    labels = df["label"].values

    tokenizer = create_tokenizer(texts, vocab_size)

    X = texts_to_padded_sequences(tokenizer, texts, max_len)
    y = labels

    save_tokenizer(tokenizer, tokenizer_path)

    return X, y


# ---------------- TEST BLOCK ---------------- #

if __name__ == "__main__":
    DATA_PATH = "data/processed/cleaned_data.csv"
    TOKENIZER_PATH = "models/tokenizer.pkl"

    VOCAB_SIZE = 5000
    MAX_LEN = 100

    print("🔹 Running tokenizer pipeline...")

    X, y = prepare_features(DATA_PATH, VOCAB_SIZE, MAX_LEN, TOKENIZER_PATH)

    print("✅ Tokenization completed!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("Sample sequence:", X[0])