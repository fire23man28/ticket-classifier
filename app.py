from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")


class TicketIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    category: str
    confidence: float


@app.post("/predict", response_model=PredictionOut)
def predict(ticket: TicketIn):
    category = model.predict([ticket.text])[0]
    confidence = model.predict_proba([ticket.text]).max()
    return PredictionOut(category=category, confidence=round(float(confidence), 2))