# src/utils/config.py

# ---------------- PATHS ---------------- #

RAW_FAKE_PATH = "data/raw/Fake.csv"
RAW_TRUE_PATH = "data/raw/True.csv"

PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"

TOKENIZER_PATH = "models/tokenizer.pkl"

MODEL_LSTM_PATH = "models/lstm_model.h5"


# ---------------- TOKENIZER PARAMS ---------------- #

VOCAB_SIZE = 4000   # slightly reduced (less noise)
MAX_LEN = 150       # 🔥 increased


# ---------------- TRAINING PARAMS ---------------- #

BATCH_SIZE = 64
EPOCHS = 6
VALIDATION_SPLIT = 0.2


# ---------------- RANDOM STATE ---------------- #

RANDOM_STATE = 42