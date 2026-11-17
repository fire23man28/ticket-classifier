# ticket-classifier

A text classifier that predicts the category of customer support tickets.

## Setup

```bash
pip install fastapi uvicorn scikit-learn pandas joblib pytest httpx ruff
```

## Train the model

The training script reads `tickets.csv` and saves the model to `model.joblib`:

```bash
python3 train.py
```

## Run the API

```bash
uvicorn app:app --reload
```

The server starts on `http://127.0.0.1:8000`.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month"}'
```

Response:

```json
{"category": "billing", "confidence": 0.45}
```

## Run the tests and linter

```bash
pytest
ruff check .
```