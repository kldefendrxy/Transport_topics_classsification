# TG Transport Classifier (MLOps pipeline)

Классификация сообщений Telegram-каналов транспортной тематики (автотранспорт vs метро) с использованием полного MLOps-пайплайна:  
DVC → обучение → артефакты модели → FastAPI → Docker → CI/CD.

---

## 🚀 Архитектура пайплайна

```mermaid
flowchart LR
    A[Telegram messages<br>data/raw] --> B[Preprocess<br>src/preprocess.py]
    B --> C[data/processed/dataset.csv]
    C --> D[Train models<br>src/train_models.py]
    D --> E[Model artifacts<br>models/*.joblib]
    E --> F[FastAPI inference<br>api/app.py]
    F --> G[Docker container]