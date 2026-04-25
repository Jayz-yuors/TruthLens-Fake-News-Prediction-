# src/data/load_data.py

import pandas as pd
import os


def load_raw_data(fake_path: str, true_path: str) -> tuple:
    """
    Load Fake and True datasets.

    Args:
        fake_path (str): Path to Fake.csv
        true_path (str): Path to True.csv

    Returns:
        tuple: (fake_df, true_df)
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake file not found at {fake_path}")

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True file not found at {true_path}")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    return fake_df, true_df


def add_labels(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> tuple:
    """
    Add labels to datasets.

    Fake → 1
    Real → 0
    """
    fake_df = fake_df.copy()
    true_df = true_df.copy()

    fake_df["label"] = 1
    true_df["label"] = 0

    return fake_df, true_df


def merge_datasets(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fake and real datasets.
    """
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def validate_dataframe(df: pd.DataFrame):
    """
    Basic validation checks.
    """
    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in dataset")

    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in dataset")

    if df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Missing values detected")


# ---------------- TEST BLOCK ---------------- #

if __name__ == "__main__":
    FAKE_PATH = "data/raw/Fake.csv"
    TRUE_PATH = "data/raw/True.csv"

    print("🔹 Loading datasets...")

    fake_df, true_df = load_raw_data(FAKE_PATH, TRUE_PATH)

    print(f"Fake shape: {fake_df.shape}")
    print(f"True shape: {true_df.shape}")

    fake_df, true_df = add_labels(fake_df, true_df)

    df = merge_datasets(fake_df, true_df)

    validate_dataframe(df)

    print("✅ Data loaded and validated successfully!")
    print(df.head())