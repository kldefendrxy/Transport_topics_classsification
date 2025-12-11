# TG Transport Classifier  
Полный MLOps-проект для классификации сообщений Telegram-каналов транспортной тематики  
(**метро** / **автотранспорт**) с использованием DVC, FastAPI, Docker и CI/CD.

---

# 1. Архитектура пайплайна

```mermaid
flowchart TD
    A[Raw data (CSV) \ndata/raw] --> B[Preprocess<br>src/preprocess.py]
    B --> C[Processed dataset.csv<br>data/processed]
    C --> D[Training<br>src/train_models.py]
    D --> E[Model artifacts (.joblib)<br>models/]
    E --> F[FastAPI inference<br>api/app.py]
    F --> G[Docker Image]
    G --> H[CI/CD GitHub Actions<br>auto-build & push]
```

⸻

# 2. Описание задачи

Цель: определить тип сообщения
из двух классов:
	•	автотранспорт
	•	метро

Модель обучается на данных из Telegram-каналов, проходит полный процесс:

очистка → лемматизация → TF-IDF → тренировка моделей → выбор лучшей → упаковка в сервис

⸻

# 3. Репозиторий и структура

MLops_course/
│
├── api/                # FastAPI inference service
│   └── app.py
│
├── configs/            # Конфигурации (train/inference)
│   ├── train_config.yaml
│   └── inference_config.yaml
│
├── data/
│   ├── raw/            # Сырые CSV
│   └── processed/      # Обработанные данные (генерируются)
│
├── models/             # Артефакты модели (генерируются DVC)
│
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   └── train_nn.py
│
├── dvc.yaml            # Описание pipeline
└── README.md


⸻

# 4. Конфигурации

train_config.yaml
 - путь к данным
 - параметры препроцессинга
 - параметры TF-IDF
 - список моделей
 - куда сохранять артефакты

inference_config.yaml
 - путь до модели
 - путь до векторизатора
 - путь до LabelEncoder

⸻

# 5. Запуск пайплайна обучения (DVC)

1) Установить зависимости
```
pip install -r requirements.txt
```

2) Запустить весь pipeline
```
dvc repro
```

DVC:
 - возьмёт данные из data/raw
 - выполнит preprocess
 - выполнит train
 - обновит models/
 - сохранит метрики

 ## Трекинг экспериментов (MLflow)

Для отслеживания экспериментов используется MLflow.

Скрипт `src/train_models.py` при каждом запуске:

- логирует параметры моделей (из `configs/train_config.yaml`);
- записывает метрики (accuracy, precision, recall, F1);
- сохраняет артефакты моделей в локальное хранилище MLflow (`./mlruns`).

Запуск MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
После этого интерфейс доступен по адресу http://127.0.0.1:5000
⸻

# 6. FastAPI сервис (инференс)

Локальный запуск
```
uvicorn api.app:app --reload
```

Эндпоинты

/health
Проверка статуса.

Response:

{"status": "ok"}

/predict
Request:
``` json
{"text": "еду по салатовой ветке"}
```

Response:
``` json
{"class": "метро"}
```

⸻

# 7. Docker

Сборка контейнера
```
docker build -t tg_classifier -f docker/Dockerfile .
```

Запуск контейнера
```
docker run -p 8000:8000 tg_classifier
```

Документация сервиса:
```
http://0.0.0.0:8000/docs
```

⸻

# 8. CI/CD (GitHub Actions)

Файл: .github/workflows/docker.yml

Пайплайн выполняет:
 - checkout репозитория
 - сборка Docker-образа
 - логин в GitHub Container Registry
 - пуш образа

Каждый пуш в main = автоматическая сборка.

⸻

# 9. Документация

MODEL_CARD.md
Описание модели, гиперпараметров, метрик, сравнение экспериментов.

DATASET_CARD.md
Описание источников данных, объёма, сплитов, особенностей очистки.

⸻

# 10. Метрики и эксперименты

Модели, участвующие в эксперименте:

Model	Accuracy	F1
LogisticRegression	0.994	0.994
LinearSVC	0.996	0.996
NaiveBayes	0.985	0.984

Использовано:
 - 3 эксперимента (3 модели)
 - 2 версии датасета (raw / processed)

⸻

# 11. Пример использования
```
curl -X POST "http://0.0.0.0:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "На Каширском шоссе произошло ДТП"}'
```

Response:
``` json
{"class": "автотранспорт"}
```