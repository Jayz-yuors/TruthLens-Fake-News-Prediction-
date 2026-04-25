# src/data/preprocess.py

import pandas as pd
import re
import os


# ---------------- CLEAN TEXT ---------------- #

def clean_text(text: str) -> str:
    """
    Clean text by:
    - Lowercasing
    - Removing special characters
    - Removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- MAIN PREPROCESS ---------------- #

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to dataset (LSTM-friendly).
    """

    df = df.copy()

    # Drop missing values
    df = df.dropna(subset=["text"])

    # Ensure string type
    df["text"] = df["text"].astype(str)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    return df


# ---------------- SAVE ---------------- #

def save_processed_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")


# ---------------- TEST BLOCK ---------------- #

if __name__ == "__main__":
    from src.data.load_data import load_raw_data, add_labels, merge_datasets

    FAKE_PATH = "data/raw/Fake.csv"
    TRUE_PATH = "data/raw/True.csv"
    OUTPUT_PATH = "data/processed/cleaned_data.csv"

    print("🔹 Running preprocessing pipeline...")

    # Load
    fake_df, true_df = load_raw_data(FAKE_PATH, TRUE_PATH)

    # Label
    fake_df, true_df = add_labels(fake_df, true_df)

    # Merge
    df = merge_datasets(fake_df, true_df)

    # Preprocess
    df_clean = preprocess_dataframe(df)

    # Save
    save_processed_data(df_clean, OUTPUT_PATH)

    print("✅ Preprocessing completed successfully!")
    print(df_clean.head())