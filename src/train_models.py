import os
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn


# === 1. Настройка MLflow ===

# Локальное хранилище экспериментов рядом с проектом
mlflow.set_tracking_uri("file:./mlruns")

# Имя эксперимента (можно поменять)
mlflow.set_experiment("TG_Transport_Classifier")


# === 2. Загрузка конфига ===

with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
vector_cfg = cfg["vectorization"]
models_cfg = cfg["models"]
eval_cfg = cfg["evaluation"]


# === 3. Загрузка данных ===

df = pd.read_csv(paths["processed_data"])
df = df.dropna(subset=["text_lemma"])

# Кодируем категории
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])

X = df["text_lemma"].astype(str)
y = df["label"]


# === 4. Train / test split ===

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=eval_cfg["test_size"],
    stratify=y,
    random_state=eval_cfg["random_state"]
)


# === 5. TF-IDF ===

tfidf_cfg = vector_cfg["tfidf"]

vectorizer = TfidfVectorizer(
    ngram_range=tuple(tfidf_cfg["ngram_range"]),
    max_features=tfidf_cfg["max_features"],
    min_df=tfidf_cfg["min_df"],
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# === 6. Фабрика моделей ===

def build_model(name, params):
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    if name == "LinearSVC":
        return LinearSVC(**params)
    if name == "NaiveBayes":
        return MultinomialNB(**params)
    raise ValueError(f"Неизвестная модель: {name}")


# === 7. Цикл экспериментов с логированием в MLflow ===

results = []

for m in models_cfg:
    name = m["name"]
    params = m["params"]

    print(f"\n=== Обучение модели {name} ===")

    # Каждый запуск модели — отдельный MLflow run
    with mlflow.start_run(run_name=name):
        # Логируем параметры из конфига
        mlflow.log_param("model_name", name)
        for p_name, p_value in params.items():
            mlflow.log_param(p_name, p_value)

        # Строим и обучаем модель
        model = build_model(name, params)
        model.fit(X_train_vec, y_train)

        # Предсказания и метрики
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )

        # Логируем метрики в MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Логируем исходные размеры данных как параметры
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("vocabulary_size", X_train_vec.shape[1])

        # Логируем модель как артефакт MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Сохраняем в локовый список для CSV
        results.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(f"{name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")


# === 8. Сохранение таблицы с результатами ===

results_df = pd.DataFrame(results)
os.makedirs(paths["model_dir"], exist_ok=True)

results_baseline_path = os.path.join(paths["model_dir"], "results_baseline.csv")
results_summary_path = os.path.join(paths["model_dir"], "results_summary.csv")

results_df.to_csv(results_baseline_path, index=False)
results_df.to_csv(results_summary_path, index=False)

print("\nРезультаты сохранены в models/")


# === 9. Выбор лучшей модели по F1 ===

best = results_df.sort_values("f1", ascending=False).iloc[0]
print("\nЛучшая модель:")
print(best)

best_model_name = best["model"]
best_model_cfg = next(m for m in models_cfg if m["name"] == best_model_name)

# Обучаем лучшую модель на тех же данных (можно потом дообучать на всём датасете, если нужно)
best_model = build_model(best_model_name, best_model_cfg["params"])
best_model.fit(X_train_vec, y_train)


# === 10. Сохранение артефактов для инференса ===

joblib.dump(best_model, paths["model_path"])
joblib.dump(vectorizer, paths["vectorizer_path"])
joblib.dump(le, paths["labels_path"])

print("\nМодель, векторизатор и LabelEncoder сохранены.")
print("Эксперименты залогированы в MLflow (директория ./mlruns).")