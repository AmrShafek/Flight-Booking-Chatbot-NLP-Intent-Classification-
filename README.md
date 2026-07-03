# ATIS Intent Classification API

Production-ready REST API that classifies airline travel queries by intent using an LSTM trained on the ATIS dataset.

## Architecture

```
┌─────────────┐     ┌────────────────────────────────────┐     ┌──────────────┐
│  Client     │────▶│  Flask / Gunicorn (app/main.py)    │────▶│  model.keras │
│  (curl/App) │     │                                    │     │  tokenizer   │
│             │     │  ┌──────────────────────────────┐  │     │  label_enc   │
│  POST       │     │  │  clean_text() ← shared code  │  │     └──────────────┘
│  /predict   │     │  └──────────────────────────────┘  │
└─────────────┘     └────────────────────────────────────┘
                              ▲
                              │ imports
                    ┌─────────┴──────────┐
                    │  src/preprocessing │
                    │  .py               │
                    └────────────────────┘
                              ▲
                              │ also imported by
                    ┌─────────┴──────────┐
                    │  src/train.py      │
                    │  (CLI, run once)   │
                    └────────────────────┘
```

## Setup

```bash
git clone <repo-url>
cd atis-intent-api
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

## Run (development)

```bash
python -m app.main
```

## Run (production)

```bash
gunicorn --workers 2 --bind 0.0.0.0:5000 app.main:app
```

## Test

```bash
pytest tests/
```

## Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "show me flights from boston to denver"}'
```

```json
{"intent": "atis_flight", "confidence": 0.9987}
```

## Docker

```bash
docker build -t atis-intent-api .
docker run -p 5000:5000 atis-intent-api
```

## Project structure

```
atis-intent-api/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── data/                  (CSVs — gitignored)
├── models/                (trained artifacts — committed)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py   (shared clean_text)
│   └── train.py           (CLI training script)
├── app/
│   ├── __init__.py
│   └── main.py            (Flask app)
├── tests/
│   └── test_api.py        (pytest suite)
└── notebooks/
    └── atis.py            (original Colab notebook, untouched)
```
