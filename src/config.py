import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "atis_intents_train.csv"
TEST_CSV  = DATA_DIR / "atis_intents_test.csv"

MODELS_DIR      = ROOT / "models"
MODEL_PATH      = MODELS_DIR / "model.keras"
TOKENIZER_PATH  = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

MAX_WORDS = 10000
MAX_LEN   = 30
EMBEDDING_DIM = 128
LSTM_UNITS    = 128
DROPOUT_RATE  = 0.5
BATCH_SIZE    = 32
EPOCHS        = 20
VALIDATION_SPLIT = 0.1
PATIENCE      = 3

DEFAULT_PORT = 5000
