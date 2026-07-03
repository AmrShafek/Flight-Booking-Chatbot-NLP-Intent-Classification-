import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.config import (
    BATCH_SIZE, DROPOUT_RATE, EPOCHS, LABEL_ENCODER_PATH,
    LSTM_UNITS, MAX_LEN, MAX_WORDS, MODEL_PATH, MODELS_DIR,
    PATIENCE, TEST_CSV, TOKENIZER_PATH, TRAIN_CSV, VALIDATION_SPLIT,
)
from src.preprocessing import clean_text


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV, header=None, names=["label", "text"])
    test_df  = pd.read_csv(TEST_CSV,  header=None, names=["label", "text"])

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test  = le.transform(test_df["label"])

    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"]  = test_df["text"].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text_clean"])

    X_train = tokenizer.texts_to_sequences(train_df["text_clean"])
    X_test  = tokenizer.texts_to_sequences(test_df["text_clean"])
    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post")
    X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="post")

    num_classes = len(le.classes_)
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(128, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        mode="max", restore_best_weights=True,
    )
    model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, callbacks=[early_stop],
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved to         {MODEL_PATH}")
    print(f"Tokenizer saved to     {TOKENIZER_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")
    print(f"Classes: {list(le.classes_)}")


if __name__ == "__main__":
    train()
