# src/explainability/lime_explainer.py

import pickle
import numpy as np

from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------- LOAD COMPONENTS ---------------- #

def load_artifacts(model_path, tokenizer_path):
    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# ---------------- PREPROCESS SINGLE TEXT ---------------- #

def preprocess_text(text, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded


# ---------------- PREDICTION FUNCTION (FOR LIME) ---------------- #

def predict_proba(texts, model, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    preds = model.predict(padded)

    # Return [REAL_prob, FAKE_prob]
    return np.hstack((1 - preds, preds))


# ---------------- MAIN PREDICTION FUNCTION ---------------- #

def predict_text(text, model, tokenizer, max_len):
    processed = preprocess_text(text, tokenizer, max_len)
    pred = model.predict(processed)[0][0]

    label = "FAKE" if pred > 0.5 else "REAL"

    # 🔥 FIXED CONFIDENCE LOGIC
    confidence = pred if pred > 0.5 else (1 - pred)

    return label, confidence


# ---------------- LIME EXPLAINER ---------------- #

def explain_text(text, model, tokenizer, max_len):

    class_names = ["REAL", "FAKE"]

    explainer = LimeTextExplainer(class_names=class_names)

    explanation = explainer.explain_instance(
        text,
        lambda x: predict_proba(x, model, tokenizer, max_len),
        num_features=10
    )

    return explanation


# ---------------- DISPLAY EXPLANATION ---------------- #

def display_explanation(explanation):

    print("\n🔍 Top contributing words:\n")

    for word, score in explanation.as_list():
        label = "FAKE" if score > 0 else "REAL"
        print(f"{word:15} → {score:.4f} ({label})")


# ---------------- TEST BLOCK ---------------- #

if __name__ == "__main__":

    MODEL_PATH = "models/lstm_model.h5"
    TOKENIZER_PATH = "models/tokenizer.pkl"
    MAX_LEN = 100

    print("🔹 Loading model and tokenizer...")

    model, tokenizer = load_artifacts(MODEL_PATH, TOKENIZER_PATH)

    sample_text = input("\nEnter news text:\n> ")

    # 🔥 FIXED PREDICTION CALL
    label, confidence = predict_text(sample_text, model, tokenizer, MAX_LEN)

    print(f"\n🧠 Prediction: {label}")
    print(f"📊 Confidence: {confidence:.4f}")

    # Explain
    explanation = explain_text(sample_text, model, tokenizer, MAX_LEN)

    display_explanation(explanation)