# src/models/lstm_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_lstm_model(vocab_size: int, max_len: int):

    model = Sequential()

    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_len
    ))

    # 🔥 BACK TO SIMPLE LSTM (STABLE)
    model.add(LSTM(64))

    # ✅ Moderate dropout (NOT aggressive)
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def print_model_summary(model):
    print("\n🔹 Model Summary:")
    model.summary()