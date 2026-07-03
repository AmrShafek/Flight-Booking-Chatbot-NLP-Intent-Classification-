import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import LABEL_ENCODER_PATH, MAX_LEN, MODEL_PATH, TOKENIZER_PATH
from src.preprocessing import clean_text

app = Flask(__name__)

_model = load_model(MODEL_PATH)
_tokenizer = pickle.loads(TOKENIZER_PATH.read_bytes())
_label_encoder = pickle.loads(LABEL_ENCODER_PATH.read_bytes())


@app.route("/", methods=["GET"])
def home():
    return "ATIS Intent Classification API"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "Missing or empty 'text' field"}), 400

    text_clean = clean_text(text)
    seq = _tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    preds = _model.predict(padded, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    intent = str(_label_encoder.inverse_transform([class_idx])[0])
    return jsonify({"intent": intent, "confidence": confidence})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
