# ticket-classifier
![CI](https://github.com/fire23man28/ticket-classifier/actions/workflows/ci.yml/badge.svg)
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

## Results

The baseline model (TF-IDF with logistic regression) reaches **0.27 accuracy** on a held-out test split of 11 tickets.

This number is low, and that's expected: the dataset has 53 hand-written examples across four categories. The test split is too small for the accuracy to be reliable, so treat it as a starting point, not a measure of quality.

## Next steps

- Collect more tickets, especially for the `account` and `technical` categories, where the baseline currently scores zero.
- Run an error analysis to see which tickets get confused and why.
- Compare the baseline against an LLM call (LLaMA 3 through Ollama) on the same test set.