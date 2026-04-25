# src/training/train.py

import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping

from src.features.tokenizer import prepare_features
from src.models.lstm_model import build_lstm_model, print_model_summary


# ---------------- CONFIG ---------------- #

DATA_PATH = "data/processed/cleaned_data.csv"
TOKENIZER_PATH = "models/tokenizer.pkl"
MODEL_PATH = "models/lstm_model.h5"
METRICS_PATH = "reports/metrics/metrics.json"

VOCAB_SIZE = 5000   # 🔥 restore original (important)
MAX_LEN = 100       # 🔥 restore original (important)

TEST_SIZE = 0.2
RANDOM_STATE = 42

EPOCHS = 5
BATCH_SIZE = 64


# ---------------- TRAINING PIPELINE ---------------- #

def train():
    print("🔹 Preparing features...")

    X, y = prepare_features(DATA_PATH, VOCAB_SIZE, MAX_LEN, TOKENIZER_PATH)

    print("🔹 Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n🔹 Computing class weights...")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        0: class_weights[0],  # REAL
        1: class_weights[1]   # FAKE
    }

    print("Class Weights:", class_weights)

    print("\n🔹 Building model...")

    model = build_lstm_model(VOCAB_SIZE, MAX_LEN)

    print_model_summary(model)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    print("\n🔹 Training model...")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weights,  # 🔥 MAIN FIX
        verbose=1
    )

    print("\n🔹 Evaluating model...")

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    save_model(model)
    save_metrics(accuracy, precision, recall, f1)


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


def save_metrics(accuracy, precision, recall, f1):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()