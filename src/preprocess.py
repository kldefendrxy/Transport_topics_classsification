import os, re, string
import numpy as np
import pandas as pd
import nltk
from tqdm.auto import tqdm
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
tqdm.pandas()

# --- загрузка ресурсов NLTK ---
nltk.download('punkt')
nltk.download('stopwords')

# --- инициализация инструментов ---
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))

def clean_text(text: str) -> str:
    """Удаление ссылок, пунктуации, чисел и приведение к нижнему регистру"""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)      # ссылки
    text = re.sub(r"@\w+|#\w+", " ", text)            # упоминания, хэштеги
    text = re.sub(r"[^а-яёa-z\s]", " ", text)         # только буквы
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text: str) -> str:
    """Токенизация + лемматизация с pymorphy3"""
    tokens = word_tokenize(text, language="russian")
    lemmas = []
    for token in tokens:
        if token not in russian_stopwords and token not in string.punctuation:
            lemmas.append(morph.parse(token)[0].normal_form)
    return " ".join(lemmas)

def preprocess_all(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(raw_dir, fname), sep=";")
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # очистка и лемматизация
    data["text_clean"] = data["text"].astype(str).progress_apply(clean_text)
    data["text_lemma"] = data["text_clean"].progress_apply(lemmatize_text)

    # сохраняем итоговый датасет
    out_path = os.path.join(out_dir, "dataset.csv")
    data.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path} ({len(data)} строк)")

if __name__ == "__main__":
    preprocess_all()