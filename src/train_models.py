import numpy as np
import pandas as pd
import joblib
import os, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# === загрузка данных ===
data = pd.read_csv("data/processed/dataset.csv")
data = data.dropna(subset=["text_lemma"])

# кодер для категорий
le = LabelEncoder()
data["category_encoded"] = le.fit_transform(data["category"])

# === подготовка признаков ===
y = data["category_encoded"]
X = data["text_lemma"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === векторизация TF-IDF ===
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === модели ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0),
    "LinearSVC": LinearSVC(),
    "NaiveBayes": MultinomialNB()
}

results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    print(f"{name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")

# === результаты ===
results_df = pd.DataFrame(results)
results_df.to_csv("models/results_baseline.csv", index=False)
print("\nРезультаты сохранены в models/results_baseline.csv")

results_df = pd.DataFrame(results)

print(results_df)

# Сохраняем в CSV
results_df.to_csv("models/results_summary.csv", index=False)

# Выводим лучшую модель
best = results_df.sort_values("f1", ascending=False).iloc[0]
print("\nЛучшая модель:")
print(best)

joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(models[best["model"]], "models/model.joblib")
joblib.dump(le, "models/labels.joblib")
print("Модель и векторизатор сохранены.")