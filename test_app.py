from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_predict_returns_category_and_confidence():
    response = client.post("/predict", json={"text": "I was charged twice this month"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in {"billing", "technical", "account", "delivery"}
    assert 0 <= data["confidence"] <= 1


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422