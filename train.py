import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_csv("tickets.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
)

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))

joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")