from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/model.joblib")
labels = joblib.load("models/labels.joblib")

class Message(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(msg: Message):
    X = vectorizer.transform([msg.text])
    pred = model.predict(X)[0]
    return {"class": labels.inverse_transform([pred])[0]}