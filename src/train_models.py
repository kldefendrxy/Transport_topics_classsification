import yaml
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder



cfg = yaml.safe_load(open("configs/train_config.yaml"))

paths = cfg["paths"]
vector_cfg = cfg["vectorization"]
models_cfg = cfg["models"]
eval_cfg = cfg["evaluation"]



df = pd.read_csv(paths["processed_data"])
df = df.dropna(subset=["text_lemma"])

# Кодируем категории
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])

X = df["text_lemma"].astype(str)
y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=eval_cfg["test_size"],
    stratify=y,
    random_state=eval_cfg["random_state"]
)



tfidf_cfg = vector_cfg["tfidf"]

vectorizer = TfidfVectorizer(
    ngram_range=tuple(tfidf_cfg["ngram_range"]),
    max_features=tfidf_cfg["max_features"],
    min_df=tfidf_cfg["min_df"],
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)



def build_model(name, params):
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    if name == "LinearSVC":
        return LinearSVC(**params)
    if name == "NaiveBayes":
        return MultinomialNB(**params)
    raise ValueError(f"Неизвестная модель: {name}")



results = []

for m in models_cfg:
    name = m["name"]
    params = m["params"]

    print(f"\n=== Обучение модели {name} ===")
    model = build_model(name, params)
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



results_df = pd.DataFrame(results)
os.makedirs(paths["model_dir"], exist_ok=True)

results_df.to_csv(os.path.join(paths["model_dir"], "results_baseline.csv"), index=False)
results_df.to_csv(os.path.join(paths["model_dir"], "results_summary.csv"), index=False)

print("\nРезультаты сохранены в models/")

best = results_df.sort_values("f1", ascending=False).iloc[0]
print("\nЛучшая модель:")
print(best)



best_model_name = best["model"]
best_model_cfg = next(m for m in models_cfg if m["name"] == best_model_name)

best_model = build_model(best_model_name, best_model_cfg["params"])
best_model.fit(X_train_vec, y_train)



joblib.dump(best_model, paths["model_path"])
joblib.dump(vectorizer, paths["vectorizer_path"])
joblib.dump(le, paths["labels_path"])

print("\nМодель, векторизатор и LabelEncoder сохранены.")