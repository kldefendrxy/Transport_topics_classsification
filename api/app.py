from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yaml
import os


config = yaml.safe_load(open("configs/inference_config.yaml"))

MODEL_PATH = config["model"]["model_path"]
VECTORIZER_PATH = config["model"]["vectorizer_path"]
LABELS_PATH = config["model"]["labels_path"]


vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)
labels_encoder = joblib.load(LABELS_PATH)


app = FastAPI(title="Telegram Transport Classifier",
              description="Классификация сообщений транспортных Telegram-каналов",
              version="1.0.0")


class Message(BaseModel):
    text: str



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict")
def predict(msg: Message):
    """Основной эндпоинт инференса"""

    text = msg.text

    # векторизация
    X = vectorizer.transform([text])

    # предсказание класса
    pred = model.predict(X)[0]

    # вероятность, если поддерживается
    try:
        prob = float(max(model.predict_proba(X)[0]))
    except:
        prob = None

    # декодирование класса
    decoded = labels_encoder.inverse_transform([pred])[0]

    return {
        "class": decoded,
        "probability": prob
    }